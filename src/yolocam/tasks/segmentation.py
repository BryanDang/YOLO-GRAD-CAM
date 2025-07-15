"""Segmentation task implementation."""

from typing import Dict, Any, Optional, Callable, Tuple, List
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
    
    def create_cam_target_function(self, **kwargs) -> Callable:
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
    
    @property
    def required_ground_truth_format(self) -> str:
        """Expected ground truth file format."""
        return 'mask_png'
    
    @property
    def supported_metrics(self) -> List[str]:
        """List of performance metrics supported by this task."""
        return ['iou', 'dice', 'pixel_accuracy']
    
    def visualize_results(self, 
                         image_path: str,
                         prediction: Any = None,
                         ground_truth: Any = None,
                         cam_output: np.ndarray = None,
                         score: Optional[float] = None,
                         title: str = "Segmentation Results",
                         **kwargs) -> np.ndarray:
        """Create segmentation-specific visualization."""
        import matplotlib.pyplot as plt
        from PIL import Image
        import matplotlib.patches as patches
        
        # Load original image
        if isinstance(image_path, str):
            image = np.array(Image.open(image_path).convert('RGB'))
        else:
            image = image_path
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 4, figsize=self.config.figure_size)
        fig.suptitle(title, fontsize=self.config.font_size + 2)
        
        # 1. Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 2. Ground truth mask
        if ground_truth is not None:
            if hasattr(ground_truth, 'cpu'):
                gt_mask = ground_truth.cpu().numpy()
            else:
                gt_mask = np.array(ground_truth)
            
            if gt_mask.ndim == 3:
                gt_mask = gt_mask[0]
            
            axes[1].imshow(image)
            axes[1].imshow(gt_mask, alpha=0.5, cmap='Blues')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
        else:
            axes[1].text(0.5, 0.5, 'No Ground Truth', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].axis('off')
        
        # 3. Prediction mask
        if prediction is not None:
            # Extract masks from YOLO results
            if hasattr(prediction, 'masks') and prediction.masks is not None:
                pred_mask = prediction.masks.data[0].cpu().numpy()
            else:
                pred_mask = np.zeros_like(image[:,:,0])
            
            axes[2].imshow(image)
            axes[2].imshow(pred_mask, alpha=0.5, cmap='Reds')
            title_text = 'Prediction'
            if score is not None:
                title_text += f'\nIoU: {score:.3f}'
            axes[2].set_title(title_text)
            axes[2].axis('off')
        else:
            axes[2].text(0.5, 0.5, 'No Prediction', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].axis('off')
        
        # 4. CAM overlay
        if cam_output is not None:
            # Resize CAM to match image size
            from scipy.ndimage import zoom
            h, w = image.shape[:2]
            cam_h, cam_w = cam_output.shape
            
            if cam_h != h or cam_w != w:
                zoom_h = h / cam_h
                zoom_w = w / cam_w
                cam_resized = zoom(cam_output, (zoom_h, zoom_w))
            else:
                cam_resized = cam_output
            
            axes[3].imshow(image)
            axes[3].imshow(cam_resized, alpha=self.config.cam_alpha, 
                          cmap=self.config.colormap)
            axes[3].set_title('Grad-CAM')
            axes[3].axis('off')
        else:
            axes[3].text(0.5, 0.5, 'No CAM', 
                        ha='center', va='center', transform=axes[3].transAxes)
            axes[3].axis('off')
        
        plt.tight_layout()
        
        # Convert to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return buf