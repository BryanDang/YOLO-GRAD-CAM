"""Segmentation task implementation."""

from typing import Dict, Any, Optional, Callable, Tuple
import numpy as np
import torch
from pathlib import Path

from .base_task import BaseTask
from ..core.registry import register_yolo_task
from ..cam.gradcam import YOLOCAMTarget


@register_yolo_task('segmentation')
class SegmentationTask(BaseTask):
    """Task implementation for YOLO segmentation models."""
    
    def compute_performance_metric(self, prediction: Any, ground_truth: Any) -> float:
        """Compute IoU for segmentation."""
        # Convert to numpy if needed
        if isinstance(prediction, torch.Tensor):
            pred_mask = prediction.cpu().numpy()
        else:
            pred_mask = np.array(prediction)
            
        if isinstance(ground_truth, torch.Tensor):
            gt_mask = ground_truth.cpu().numpy()
        else:
            gt_mask = np.array(ground_truth)
        
        # Binarize masks
        pred_mask = (pred_mask > self.config.task_configs.get('mask_threshold', 0.5)).astype(float)
        gt_mask = (gt_mask > 0.5).astype(float)
        
        # Compute IoU
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            return 0.0
            
        return float(intersection / union)
    
    def create_cam_target_function(self, image_path: str, 
                                  mask_path: Optional[str] = None) -> Callable:
        """Create CAM target for segmentation."""
        return YOLOCAMTarget(
            task='segmentation',
            class_idx=None  # Target all classes for now
        )
    
    def process_model_output(self, output: Any) -> Any:
        """Process raw model output for segmentation."""
        # Handle different output formats
        if isinstance(output, (list, tuple)):
            # YOLO typically returns [boxes, masks] for segmentation
            if len(output) > 1:
                return output[1]  # Return masks
            else:
                return output[0]
        return output
    
    def load_ground_truth(self, gt_path: str) -> Any:
        """Load ground truth mask."""
        from PIL import Image
        import torchvision.transforms as T
        
        # Load mask
        mask = Image.open(gt_path).convert('L')
        
        # Resize to model input size
        transform = T.Compose([
            T.Resize((self.config.model_input_size, self.config.model_input_size)),
            T.ToTensor(),
        ])
        
        return transform(mask)
    
    def get_task_specific_metrics(self) -> Dict[str, Callable]:
        """Get segmentation-specific metrics."""
        return {
            'iou': self.compute_performance_metric,
            'dice': self._compute_dice,
            'pixel_accuracy': self._compute_pixel_accuracy,
        }
    
    def _compute_dice(self, prediction: Any, ground_truth: Any) -> float:
        """Compute Dice coefficient."""
        # Convert to numpy
        if isinstance(prediction, torch.Tensor):
            pred_mask = prediction.cpu().numpy()
        else:
            pred_mask = np.array(prediction)
            
        if isinstance(ground_truth, torch.Tensor):
            gt_mask = ground_truth.cpu().numpy()
        else:
            gt_mask = np.array(ground_truth)
        
        # Binarize
        pred_mask = (pred_mask > self.config.task_configs.get('mask_threshold', 0.5))
        gt_mask = (gt_mask > 0.5)
        
        # Compute Dice
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        return float(2 * intersection / (pred_mask.sum() + gt_mask.sum() + 1e-8))
    
    def _compute_pixel_accuracy(self, prediction: Any, ground_truth: Any) -> float:
        """Compute pixel accuracy."""
        # Convert to numpy
        if isinstance(prediction, torch.Tensor):
            pred_mask = prediction.cpu().numpy()
        else:
            pred_mask = np.array(prediction)
            
        if isinstance(ground_truth, torch.Tensor):
            gt_mask = ground_truth.cpu().numpy()
        else:
            gt_mask = np.array(ground_truth)
        
        # Binarize
        pred_mask = (pred_mask > self.config.task_configs.get('mask_threshold', 0.5))
        gt_mask = (gt_mask > 0.5)
        
        # Compute accuracy
        correct = (pred_mask == gt_mask).sum()
        total = gt_mask.size
        
        return float(correct / total) if total > 0 else 0.0
    
    @property
    def task_name(self) -> str:
        """Task identifier."""
        return 'segmentation'