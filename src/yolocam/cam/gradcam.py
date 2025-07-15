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
        
        # Initialize GradCAM
        self.cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers,
            use_cuda=self.device == 'cuda'
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
        # Use default target if none provided
        if target_function is None:
            target_function = ClassifierOutputTarget(0)
        
        # Generate CAM
        cam_output = self.cam(input_tensor, targets=[target_function])
        
        return cam_output
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'cam'):
            self.cam.__exit__(None, None, None)


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
        
        # Target specific class or average
        if self.class_idx is not None:
            return seg_output[:, self.class_idx].mean()
        else:
            return seg_output.mean()
    
    def _detection_target(self, output: Any) -> torch.Tensor:
        """Target for detection task."""
        # Handle different output formats
        if isinstance(output, (list, tuple)):
            det_output = output[0]
        else:
            det_output = output
        
        # Target specific box confidence or average
        if self.box_idx is not None:
            return det_output[:, self.box_idx, 4]  # Confidence score
        else:
            return det_output[:, :, 4].max()  # Max confidence
    
    def _classification_target(self, output: Any) -> torch.Tensor:
        """Target for classification task."""
        if isinstance(output, (list, tuple)):
            cls_output = output[0]
        else:
            cls_output = output
        
        # Target specific class or max
        if self.class_idx is not None:
            return cls_output[:, self.class_idx]
        else:
            return cls_output.max()