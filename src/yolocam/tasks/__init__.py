"""Task-specific implementations for different YOLO use cases."""

from .base_task import BaseTask
from .detection import DetectionTask
from .segmentation import SegmentationTask
from .classification import ClassificationTask

__all__ = [
    "BaseTask",
    "DetectionTask", 
    "SegmentationTask",
    "ClassificationTask",
]