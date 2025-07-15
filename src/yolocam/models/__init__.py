"""YOLO model implementations and utilities."""

from .base_model import BaseYOLOModel
from .yolo_wrapper import YOLOv8Model, detect_model_version

__all__ = [
    "BaseYOLOModel",
    "YOLOv8Model",
    "detect_model_version",
]