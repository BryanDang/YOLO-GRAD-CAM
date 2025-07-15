"""YOLOv8 wrapper for Grad-CAM analysis using Ultralytics."""

from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from pathlib import Path

from .base_model import BaseYOLOModel
from ..core.registry import register_yolo_model


@register_yolo_model('yolov8', pattern=r'.*yolo.*v?8.*')
class YOLOv8Model(BaseYOLOModel):
    """Wrapper for Ultralytics YOLOv8 models to enable Grad-CAM analysis."""
    
    def __init__(self, model_path: str, config: 'YoloCAMConfig'):
        """Initialize YOLOv8 model wrapper."""
        super().__init__(model_path, config)
        self._yolo_model = None
    
    def load_model(self, model_path: str) -> nn.Module:
        """Load YOLOv8 model via Ultralytics and return PyTorch module."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics is not installed. "
                "Install with: pip install ultralytics"
            )
        
        # Load YOLO model from Ultralytics
        self._yolo_model = YOLO(model_path)
        
        # Get the underlying PyTorch model for Grad-CAM
        # Ultralytics structure: YOLO.model is the PyTorch model
        pytorch_model = self._yolo_model.model
        pytorch_model.eval()
        
        return pytorch_model.to(self.device)
    
    def get_target_layers(self, component: str = 'backbone') -> List[nn.Module]:
        """Get target layers from Ultralytics model for Grad-CAM hooks.
        
        YOLOv8 structure (accessed via model.model):
        - model.0-9: Backbone layers (CSPDarknet)
        - model.10-20: Neck layers (PANet) 
        - model.21-22: Head layers (Detect/Segment)
        """
        if not hasattr(self, '_pytorch_model') or self._pytorch_model is None:
            self._pytorch_model = self.load_model(self.model_path)
        
        # Access the actual model layers
        model = self._pytorch_model.model
        
        if component == 'backbone':
            # Last backbone layer - typically model.9 in YOLOv8
            try:
                return [model[9]]
            except:
                # Fallback to any late backbone layer
                for i in range(9, 5, -1):
                    if i < len(model):
                        return [model[i]]
                        
        elif component == 'neck':
            # SPPF layer in neck - typically model.12
            try:
                return [model[12]]
            except:
                # Fallback to middle neck layer
                for i in range(15, 10, -1):
                    if i < len(model):
                        return [model[i]]
                        
        elif component == 'head':
            # Detection/Segmentation head - typically model.22
            try:
                # Try to get the last layer
                if hasattr(model, '22'):
                    return [model[22]]
                elif hasattr(model, '21'):
                    return [model[21]]
                else:
                    # Get the last sequential layer
                    return [model[-1]]
            except:
                return [model[-1]]
                
        else:
            raise ValueError(f"Unknown component: {component}. Use 'backbone', 'neck', or 'head'")
            
        # Default fallback
        return [model[9]]  # Safe default to backbone
    
    def get_inference_model(self) -> Any:
        """Get Ultralytics YOLO model for inference."""
        if not hasattr(self, '_yolo_model'):
            self.load_model(self.model_path)
        return self._yolo_model
    
    def preprocess_input(self, image_path: str) -> torch.Tensor:
        """Preprocess image for Grad-CAM analysis."""
        from PIL import Image
        import torchvision.transforms as T
        from torchvision.transforms import functional as F
        import numpy as np
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Get original dimensions
        orig_w, orig_h = image.size
        
        # YOLO uses letterbox preprocessing - we'll do a simplified version
        # For Grad-CAM, we can use standard resizing as the CAM will be resized anyway
        target_size = self.config.model_input_size
        
        # Resize image to model input size
        image = F.resize(image, (target_size, target_size))
        
        # Convert to tensor and normalize (YOLO expects 0-1 range)
        tensor = F.to_tensor(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def detect_task(self) -> str:
        """Auto-detect task from Ultralytics model."""
        # First try filename detection (doesn't require loading model)
        model_name = str(self.model_path).lower()
        
        if 'seg' in model_name:
            return 'segmentation'  # Keep consistent with our task names
        elif 'cls' in model_name:
            return 'classification'
        elif 'pose' in model_name:
            return 'pose'
        else:
            # If can't detect from filename, load model to check
            if not hasattr(self, '_yolo_model'):
                # Temporarily load just to detect task
                try:
                    from ultralytics import YOLO
                    temp_model = YOLO(self.model_path)
                    task = getattr(temp_model, 'task', 'detection')
                    # Map Ultralytics task names to our names
                    task_map = {
                        'segment': 'segmentation',
                        'classify': 'classification',
                        'detect': 'detection'
                    }
                    return task_map.get(task, task)
                except:
                    return 'detection'  # Default fallback
                    
            # If model already loaded
            if hasattr(self, '_yolo_model') and hasattr(self._yolo_model, 'task'):
                task = self._yolo_model.task
                # Map Ultralytics task names to our names
                task_map = {
                    'segment': 'segmentation',
                    'classify': 'classification',
                    'detect': 'detection'
                }
                return task_map.get(task, task)
                
            return 'detection'
    
    @property
    def supported_tasks(self) -> List[str]:
        """YOLOv8 supports all major tasks."""
        return ['detection', 'segmentation', 'classification', 'pose']
    
    @property
    def version(self) -> str:
        """Model version identifier."""
        return 'yolov8'


def detect_model_version(model_path: str) -> str:
    """Detect YOLO model version from path or file."""
    path_str = str(model_path).lower()
    
    # Check for version patterns
    if 'v10' in path_str or 'yolo10' in path_str:
        return 'yolov10'
    elif 'v9' in path_str or 'yolo9' in path_str:
        return 'yolov9'
    elif 'v8' in path_str or 'yolo8' in path_str:
        return 'yolov8'
    elif 'v5' in path_str or 'yolo5' in path_str:
        return 'yolov5'
    
    # Default to v8 for now
    return 'yolov8'