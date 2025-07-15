"""Task-specific implementations for different YOLO use cases."""

from .base_task import BaseTask
from .segmentation import SegmentationTask

__all__ = [
    "BaseTask",
    "SegmentationTask",
]