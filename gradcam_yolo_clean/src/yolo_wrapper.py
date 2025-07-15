"""YOLOv8 wrapper that exposes PyTorch layers for Grad-CAM."""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image


class YOLOWrapper:
    """Wrapper for YOLOv8 that provides access to PyTorch layers."""
    
    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'auto'):
        """
        Initialize YOLO wrapper.
        
        Args:
            model_path: Path to YOLO model or model name
            device: Device to run on ('auto', 'cpu', 'cuda')
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load YOLO model
        self.yolo = YOLO(model_path)
        
        # Get the underlying PyTorch model
        self.model = self.yolo.model.to(self.device)
        self.model.eval()
        
        # Store model info
        self.model_path = model_path
        self.task = self._detect_task()
        
    def _detect_task(self) -> str:
        """Detect the task type from model."""
        model_name = str(self.model_path).lower()
        
        if 'seg' in model_name:
            return 'segmentation'
        elif 'cls' in model_name:
            return 'classification'
        elif 'pose' in model_name:
            return 'pose'
        else:
            return 'detection'
    
    def predict(self, image: Any, **kwargs) -> Any:
        """
        Run YOLO prediction on image.
        
        Args:
            image: PIL Image, numpy array, or path to image
            **kwargs: Additional arguments for YOLO predict
            
        Returns:
            YOLO Results object
        """
        return self.yolo(image, **kwargs)
    
    def get_target_layer(self, layer_type: str = 'backbone') -> nn.Module:
        """
        Get target layer for Grad-CAM.
        
        Args:
            layer_type: Type of layer ('backbone', 'neck', 'head')
            
        Returns:
            PyTorch module for Grad-CAM
        """
        # Access the sequential model
        sequential_model = self.model.model
        
        if layer_type == 'backbone':
            # Last backbone layer (typically layer 9)
            return sequential_model[9]
        elif layer_type == 'neck':
            # SPPF layer (typically layer 12)
            return sequential_model[12]
        elif layer_type == 'head':
            # Detection head (last layer)
            if self.task == 'segmentation':
                # For segmentation, use the last conv before segment head
                return sequential_model[21]
            else:
                # For detection, use the last layer
                return sequential_model[-1]
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
    def preprocess_image(self, image_path: str, size: int = 640) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image_path: Path to image
            size: Input size for model
            
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize
        image = image.resize((size, size))
        
        # Convert to tensor and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        return self.model(x)