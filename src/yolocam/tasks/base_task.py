"""Base task interface for different YOLO use cases."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
from pathlib import Path


class BaseTask(ABC):
    """Abstract base class for YOLO task implementations.
    
    This class defines the interface that all task implementations
    must follow to be compatible with the YoloCAM library.
    """
    
    def __init__(self, config: 'YoloCAMConfig'):
        """Initialize the task handler.
        
        Args:
            config: Configuration object with task-specific settings
        """
        self.config = config
        self.task_config = config.task_configs.get(self.task_name, {})
    
    @abstractmethod
    def compute_performance_metric(self, 
                                 prediction: Any, 
                                 ground_truth: Any,
                                 **kwargs) -> float:
        """Compute task-specific performance metric.
        
        Args:
            prediction: Model prediction output
            ground_truth: Ground truth annotation
            **kwargs: Additional task-specific parameters
            
        Returns:
            Performance score (higher is better)
            
        Raises:
            ValueError: If inputs are incompatible
        """
        pass
    
    @abstractmethod
    def create_cam_target_function(self, **kwargs) -> Callable:
        """Create CAM target function for this task.
        
        Args:
            **kwargs: Task-specific parameters for target function
            
        Returns:
            Function that computes target values from model output
        """
        pass
    
    @abstractmethod
    def load_ground_truth(self, path: str) -> Any:
        """Load ground truth data for this task.
        
        Args:
            path: Path to ground truth file
            
        Returns:
            Loaded ground truth in task-specific format
            
        Raises:
            FileNotFoundError: If ground truth file doesn't exist
            ValueError: If file format is unsupported
        """
        pass
    
    @abstractmethod
    def visualize_results(self, 
                         image: np.ndarray,
                         prediction: Any,
                         ground_truth: Any,
                         cam_output: np.ndarray,
                         **kwargs) -> np.ndarray:
        """Create task-specific visualization.
        
        Args:
            image: Original input image
            prediction: Model prediction
            ground_truth: Ground truth annotation
            cam_output: CAM heatmap
            **kwargs: Additional visualization parameters
            
        Returns:
            Visualization image as numpy array
        """
        pass
    
    @property
    @abstractmethod
    def task_name(self) -> str:
        """Task identifier.
        
        Returns:
            Unique task name (e.g., 'detection', 'segmentation')
        """
        pass
    
    @property
    @abstractmethod
    def supported_metrics(self) -> List[str]:
        """List of performance metrics supported by this task.
        
        Returns:
            List of metric names available for this task
        """
        pass
    
    @property
    @abstractmethod
    def required_ground_truth_format(self) -> str:
        """Expected ground truth file format.
        
        Returns:
            File format description (e.g., 'mask_png', 'yolo_txt', 'coco_json')
        """
        pass
    
    def analyze_performance(self, 
                          model_handler: 'BaseYOLOModel',
                          image_dir: str, 
                          ground_truth_dir: str,
                          **kwargs) -> List[Dict[str, Any]]:
        """Analyze model performance on a dataset.
        
        Args:
            model_handler: Model implementation instance
            image_dir: Directory containing images
            ground_truth_dir: Directory containing ground truth
            **kwargs: Additional analysis parameters
            
        Returns:
            List of dictionaries with performance data for each image
        """
        results = []
        image_dir = Path(image_dir)
        ground_truth_dir = Path(ground_truth_dir)
        
        # Get all valid image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in image_dir.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        for image_file in image_files:
            try:
                # Find corresponding ground truth file
                gt_file = self._find_ground_truth_file(
                    image_file, ground_truth_dir
                )
                if gt_file is None:
                    continue
                
                # Load ground truth
                ground_truth = self.load_ground_truth(str(gt_file))
                
                # Get model prediction
                prediction = self._get_model_prediction(
                    model_handler, str(image_file)
                )
                
                # Compute performance metric
                score = self.compute_performance_metric(
                    prediction, ground_truth, **kwargs
                )
                
                results.append({
                    'filename': image_file.name,
                    'image_path': str(image_file),
                    'ground_truth_path': str(gt_file),
                    'score': score,
                    'prediction': prediction,
                    'ground_truth': ground_truth,
                })
                
            except Exception as e:
                print(f"Warning: Failed to process {image_file.name}: {e}")
                continue
        
        # Sort by performance score
        results.sort(key=lambda x: x['score'])
        return results
    
    def _find_ground_truth_file(self, 
                               image_file: Path, 
                               ground_truth_dir: Path) -> Optional[Path]:
        """Find the corresponding ground truth file for an image.
        
        Args:
            image_file: Path to image file
            ground_truth_dir: Directory containing ground truth files
            
        Returns:
            Path to ground truth file or None if not found
        """
        # Try various naming conventions
        base_name = image_file.stem
        
        # Common ground truth file extensions by task
        gt_extensions = self._get_ground_truth_extensions()
        
        for ext in gt_extensions:
            gt_file = ground_truth_dir / f"{base_name}{ext}"
            if gt_file.exists():
                return gt_file
        
        return None
    
    def _get_ground_truth_extensions(self) -> List[str]:
        """Get possible ground truth file extensions for this task.
        
        Returns:
            List of file extensions to try
        """
        # Default extensions - should be overridden by specific tasks
        return ['.txt', '.json', '.xml', '.png', '.jpg']
    
    def _get_model_prediction(self, 
                            model_handler: 'BaseYOLOModel',
                            image_path: str) -> Any:
        """Get model prediction for an image.
        
        Args:
            model_handler: Model implementation instance
            image_path: Path to input image
            
        Returns:
            Model prediction in task-specific format
        """
        # Use inference model for predictions
        inference_model = model_handler.inference_model
        results = inference_model(image_path, verbose=False, 
                                imgsz=self.config.model_input_size)
        return results[0]  # Return first result
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get comprehensive task information.
        
        Returns:
            Dictionary with task metadata and capabilities
        """
        return {
            'task_name': self.task_name,
            'supported_metrics': self.supported_metrics,
            'ground_truth_format': self.required_ground_truth_format,
            'config': self.task_config,
        }