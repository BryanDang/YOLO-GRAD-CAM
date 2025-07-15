"""Grad-CAM implementation using PyTorch hooks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable, List
import cv2


class GradCAMWrapper:
    """Grad-CAM implementation with forward/backward hooks."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM wrapper.
        
        Args:
            model: PyTorch model
            target_layer: Target layer for CAM
        """
        self.model = model
        self.target_layer = target_layer
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)
        
    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output.detach()
        
    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook to save gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor: torch.Tensor, 
                     target_class: Optional[int] = None,
                     target_fn: Optional[Callable] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor [1, 3, H, W]
            target_class: Target class index (for classification)
            target_fn: Custom target function for complex models
            
        Returns:
            CAM heatmap as numpy array [H, W]
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Handle different output types
        if target_fn is not None:
            # Use custom target function
            target = target_fn(output)
        elif isinstance(output, (list, tuple)):
            # YOLO-style output
            if len(output) > 0 and hasattr(output[0], 'shape'):
                # Detection output: [batch, num_predictions, num_classes + 5]
                # Use max confidence score as target
                pred_tensor = output[0] if torch.is_tensor(output[0]) else output[0].pred
                if len(pred_tensor.shape) == 3:
                    # Get confidence scores
                    confidences = pred_tensor[0, :, 4]  # objectness score
                    target = confidences.max()
                else:
                    target = pred_tensor.mean()
            else:
                # Fallback
                target = output[0].mean()
        else:
            # Standard classification output
            if target_class is not None:
                target = output[0, target_class]
            else:
                target = output.max()
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # Generate CAM
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients")
        
        # Pool gradients across spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        
        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # ReLU (only positive influence)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Convert to numpy
        cam = cam.cpu().numpy()
        
        # Resize to input size
        input_size = input_tensor.shape[-2:]  # (H, W)
        cam = cv2.resize(cam, (input_size[1], input_size[0]))
        
        return cam
    
    def overlay_cam_on_image(self, image: np.ndarray, cam: np.ndarray, 
                            alpha: float = 0.5) -> np.ndarray:
        """
        Overlay CAM heatmap on image.
        
        Args:
            image: Original image [H, W, 3] in range [0, 1]
            cam: CAM heatmap [H, W] in range [0, 1]
            alpha: Blending factor
            
        Returns:
            Overlayed image [H, W, 3]
        """
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0
        
        # Ensure image is float
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        # Overlay
        overlay = (1 - alpha) * image + alpha * heatmap
        
        return overlay
    
    def __del__(self):
        """Remove hooks when object is deleted."""
        if hasattr(self, 'forward_hook'):
            self.forward_hook.remove()
        if hasattr(self, 'backward_hook'):
            self.backward_hook.remove()