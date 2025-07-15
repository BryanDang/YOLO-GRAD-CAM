"""Base model interface for YOLO implementations."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn


class BaseYOLOModel(ABC):
    """Abstract base class for YOLO model implementations.
    
    This class defines the interface that all YOLO model implementations
    must follow to be compatible with the YoloCAM library.
    """
    
    def __init__(self, model_path: str, config: 'YoloCAMConfig'):
        """Initialize the YOLO model.
        
        Args:
            model_path: Path to the model file (.pt, .yaml, etc.)
            config: Configuration object with model settings
        """
        self.model_path = model_path
        self.config = config
        self.device = self._detect_device()
        self._pytorch_model: Optional[nn.Module] = None
        self._inference_model: Optional[Any] = None
    
    def _detect_device(self) -> torch.device:
        """Detect and return the appropriate device."""
        if self.config.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.config.device)
    
    @abstractmethod
    def load_model(self, model_path: str) -> nn.Module:
        """Load the YOLO model from path and return PyTorch module.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            PyTorch model suitable for CAM analysis
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model format is unsupported
        """
        pass
    
    @abstractmethod
    def get_target_layers(self, component: str = 'backbone') -> List[nn.Module]:
        """Get recommended target layers for CAM analysis.
        
        Args:
            component: Model component ('backbone', 'neck', 'head')
            
        Returns:
            List of PyTorch modules suitable for hooking
            
        Raises:
            ValueError: If component is not supported
        """
        pass
    
    @abstractmethod
    def get_inference_model(self) -> Any:
        """Get model optimized for inference operations.
        
        Returns:
            Model object for making predictions
        """
        pass
    
    @abstractmethod
    def preprocess_input(self, image_path: str) -> torch.Tensor:
        """Preprocess input image for the model.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed tensor ready for model input
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format is unsupported
        """
        pass
    
    @abstractmethod
    def detect_task(self) -> str:
        """Auto-detect the primary task this model was trained for.
        
        Returns:
            Task name ('detection', 'segmentation', 'classification', 'pose')
            
        Raises:
            ValueError: If task cannot be determined
        """
        pass
    
    @property
    @abstractmethod
    def supported_tasks(self) -> List[str]:
        """List of tasks supported by this model version.
        
        Returns:
            List of task names this model version can handle
        """
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Model version identifier.
        
        Returns:
            Version string (e.g., 'yolov8', 'yolov9', 'yolov10')
        """
        pass
    
    @property
    def pytorch_model(self) -> nn.Module:
        """Get the PyTorch model, loading if necessary."""
        if self._pytorch_model is None:
            self._pytorch_model = self.load_model(self.model_path)
        return self._pytorch_model
    
    @property
    def inference_model(self) -> Any:
        """Get the inference model, loading if necessary."""
        if self._inference_model is None:
            self._inference_model = self.get_inference_model()
        return self._inference_model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information.
        
        Returns:
            Dictionary with model metadata and capabilities
        """
        return {
            'version': self.version,
            'supported_tasks': self.supported_tasks,
            'device': str(self.device),
            'model_path': self.model_path,
            'input_size': self.config.model_input_size,
            'primary_task': self.detect_task(),
        }
    
    def validate_compatibility(self, task: str) -> bool:
        """Check if this model supports the specified task.
        
        Args:
            task: Task name to check
            
        Returns:
            True if task is supported, False otherwise
        """
        return task in self.supported_tasks