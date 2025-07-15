"""Grad-CAM implementation wrapper for YOLO models."""

from typing import Optional, List, Callable, Any
import torch
import torch.nn as nn
import numpy as np

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False


class GradCAMWrapper:
    """Wrapper for pytorch-grad-cam library with YOLO-specific adaptations."""
    
    def __init__(self, model: nn.Module, target_layers: List[nn.Module], 
                 device: str = 'cuda'):
        """Initialize Grad-CAM wrapper.
        
        Args:
            model: PyTorch model
            target_layers: List of target layers for CAM
            device: Device to run on
        """
        if not GRADCAM_AVAILABLE:
            raise ImportError(
                "pytorch-grad-cam is not installed. "
                "Install with: pip install pytorch-grad-cam"
            )
        
        self.model = model
        self.target_layers = target_layers
        self.device = device
        
        # Enable gradients for the model
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Initialize GradCAM
        self.cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers
        )
    
    def generate_cam(self, input_tensor: torch.Tensor, 
                     target_function: Optional[Callable] = None) -> np.ndarray:
        """Generate CAM heatmap for input.
        
        Args:
            input_tensor: Input tensor [B, C, H, W]
            target_function: Custom target function for CAM
            
        Returns:
            CAM heatmap as numpy array [B, H, W]
        """
        # Clone the input tensor to avoid inference mode issues
        # This is critical for gradient computation
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Use default target if none provided
        if target_function is None:
            target_function = ClassifierOutputTarget(0)
        
        # Disable inference mode and enable gradients explicitly
        with torch.inference_mode(False):
            with torch.set_grad_enabled(True):
                # Generate CAM
                cam_output = self.cam(input_tensor, targets=[target_function])
        
        return cam_output
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'cam'):
            self.cam.__exit__(None, None, None)


class ManualGradCAM:
    """Manual Grad-CAM implementation for better control over gradients."""
    
    def __init__(self, model: nn.Module, target_layers: List[nn.Module], 
                 device: str = 'cuda'):
        """Initialize manual Grad-CAM."""
        self.model = model
        self.target_layers = target_layers
        self.device = device
        
        # Storage for activations and gradients
        self.activations = {}
        self.gradients = {}
        
        # Register hooks
        self.handles = []
        self._register_hooks()
        
        # Enable gradients
        for param in self.model.parameters():
            param.requires_grad = True
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layers."""
        for idx, layer in enumerate(self.target_layers):
            # Forward hook to capture activations
            def save_activation(module, input, output, idx=idx):
                self.activations[idx] = output.detach()
            
            # Backward hook to capture gradients
            def save_gradient(module, grad_input, grad_output, idx=idx):
                self.gradients[idx] = grad_output[0].detach()
            
            handle_forward = layer.register_forward_hook(save_activation)
            handle_backward = layer.register_backward_hook(save_gradient)
            
            self.handles.extend([handle_forward, handle_backward])
    
    def generate_cam(self, input_tensor: torch.Tensor, 
                     target_function: Callable) -> np.ndarray:
        """Generate CAM using manual implementation."""
        # Clear previous activations/gradients
        self.activations.clear()
        self.gradients.clear()
        
        # Clone input tensor to avoid inference mode issues
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Disable inference mode and enable gradients
        with torch.inference_mode(False):
            with torch.set_grad_enabled(True):
                # Forward pass
                self.model.eval()  # Keep in eval mode but with gradients
                
                # Clear gradients before forward pass
                self.model.zero_grad()
                
                # Forward pass
                output = self.model(input_tensor)
                
                # Get target score
                target_score = target_function(output)
                
                # Backward pass with gradient retention
                target_score.backward(retain_graph=True)
        
        # Generate CAM for each target layer
        cam_per_layer = []
        
        for idx in range(len(self.target_layers)):
            if idx in self.activations and idx in self.gradients:
                activation = self.activations[idx]
                gradient = self.gradients[idx]
                
                # Global average pooling on gradients
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                
                # Weighted combination of activations
                cam = (weights * activation).sum(dim=1, keepdim=True)
                
                # ReLU
                cam = torch.relu(cam)
                
                # Normalize
                cam_min = cam.min()
                cam_max = cam.max()
                if cam_max > cam_min:
                    cam = (cam - cam_min) / (cam_max - cam_min)
                
                cam_per_layer.append(cam.squeeze().cpu().numpy())
        
        # Average CAMs from all layers
        if cam_per_layer:
            final_cam = np.mean(cam_per_layer, axis=0)
        else:
            # Fallback to zeros
            final_cam = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        
        return final_cam
    
    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.handles:
            handle.remove()
    
    def __del__(self):
        """Clean up hooks."""
        self.remove_hooks()


class YOLOCAMTarget:
    """Custom CAM target for YOLO models."""
    
    def __init__(self, task: str, class_idx: Optional[int] = None,
                 box_idx: Optional[int] = None):
        """Initialize YOLO CAM target.
        
        Args:
            task: Task type ('detection', 'segmentation', etc.)
            class_idx: Target class index
            box_idx: Target box/mask index
        """
        self.task = task
        self.class_idx = class_idx
        self.box_idx = box_idx
    
    def __call__(self, model_output: Any) -> torch.Tensor:
        """Compute target activation for CAM.
        
        Args:
            model_output: Model output (format depends on task)
            
        Returns:
            Scalar tensor for gradient computation
        """
        if self.task == 'segmentation':
            return self._segmentation_target(model_output)
        elif self.task == 'detection':
            return self._detection_target(model_output)
        elif self.task == 'classification':
            return self._classification_target(model_output)
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def _segmentation_target(self, output: Any) -> torch.Tensor:
        """Target for segmentation task."""
        # Handle different output formats
        if isinstance(output, (list, tuple)):
            # YOLO typically returns [detection_output, segmentation_output]
            if len(output) > 1:
                seg_output = output[1]
            else:
                seg_output = output[0]
        else:
            seg_output = output
        
        # Ensure we have a tensor with gradients
        if not isinstance(seg_output, torch.Tensor):
            raise ValueError(f"Expected tensor, got {type(seg_output)}")
        
        # Target specific class or average
        if self.class_idx is not None and seg_output.shape[1] > self.class_idx:
            return seg_output[:, self.class_idx].mean()
        else:
            # Use mean of all channels
            return seg_output.mean()
    
    def _detection_target(self, output: Any) -> torch.Tensor:
        """Target for detection task."""
        # Handle different output formats
        if isinstance(output, (list, tuple)):
            det_output = output[0]
        else:
            det_output = output
        
        # Ensure tensor
        if not isinstance(det_output, torch.Tensor):
            raise ValueError(f"Expected tensor, got {type(det_output)}")
        
        # Target specific box confidence or average
        if det_output.dim() >= 3 and det_output.shape[-1] >= 5:
            # Standard YOLO output format
            if self.box_idx is not None and det_output.shape[1] > self.box_idx:
                return det_output[:, self.box_idx, 4]  # Confidence score
            else:
                return det_output[:, :, 4].max()  # Max confidence
        else:
            # Fallback to mean
            return det_output.mean()
    
    def _classification_target(self, output: Any) -> torch.Tensor:
        """Target for classification task."""
        if isinstance(output, (list, tuple)):
            cls_output = output[0]
        else:
            cls_output = output
        
        # Ensure tensor
        if not isinstance(cls_output, torch.Tensor):
            raise ValueError(f"Expected tensor, got {type(cls_output)}")
        
        # Target specific class or max
        if self.class_idx is not None and cls_output.shape[1] > self.class_idx:
            return cls_output[:, self.class_idx].mean()
        else:
            return cls_output.max()