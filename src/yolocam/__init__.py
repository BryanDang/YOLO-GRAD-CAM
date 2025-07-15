"""
YoloCAM: A comprehensive library for YOLO model analysis using Grad-CAM and other explainable AI techniques.

This library provides tools for analyzing YOLO models (v8, v9, v10+) across different tasks 
(detection, segmentation, classification, pose estimation) using various computer vision 
explanation techniques, primarily Grad-CAM.

Example:
    Basic usage for segmentation analysis:
    
    >>> from yolocam import YoloCAMAnalyzer
    >>> analyzer = YoloCAMAnalyzer("path/to/model.pt", task="segmentation")
    >>> results = analyzer.analyze_performance("images/", "masks/")
    >>> analyzer.visualize_results("image.jpg", "mask.png")

"""

__version__ = "0.1.0"
__author__ = "YoloCAM Contributors"
__email__ = "yolocam@example.com"
__license__ = "MIT"

# Core imports for public API
from .core.analyzer import YoloCAMAnalyzer
from .core.config import YoloCAMConfig
from .core.registry import YOLOModelRegistry, TaskRegistry

# Model imports
from .models import (
    BaseYOLOModel,
    YOLOv8Model,
    detect_model_version,
)

# Task imports  
from .tasks import (
    BaseTask,
    DetectionTask,
    SegmentationTask,
    ClassificationTask,
)

# CAM imports
from .cam import (
    GradCAMWrapper,
    EigenCAMWrapper,
)

# Utility imports
from .utils.logging import setup_logging
from .utils.validation import validate_model_path, validate_image_path

__all__ = [
    # Core classes
    "YoloCAMAnalyzer",
    "YoloCAMConfig", 
    
    # Registry
    "YOLOModelRegistry",
    "TaskRegistry",
    
    # Base classes
    "BaseYOLOModel",
    "BaseTask",
    
    # Model implementations
    "YOLOv8Model",
    "detect_model_version",
    
    # Task implementations
    "DetectionTask", 
    "SegmentationTask",
    "ClassificationTask",
    
    # CAM implementations
    "GradCAMWrapper",
    "EigenCAMWrapper",
    
    # Utilities
    "setup_logging",
    "validate_model_path",
    "validate_image_path",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

# Configure default logging
setup_logging()