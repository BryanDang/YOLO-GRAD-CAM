"""Main analyzer class for YOLO Grad-CAM analysis."""

import os
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
import numpy as np

from .config import YoloCAMConfig
from .registry import YOLOModelRegistry, TaskRegistry
from ..models.base_model import BaseYOLOModel
from ..tasks.base_task import BaseTask
from ..cam.gradcam import GradCAMWrapper
from ..utils.logging import get_logger
from ..utils.validation import validate_model_path, validate_directory


class YoloCAMAnalyzer:
    """Main class for YOLO Grad-CAM analysis.
    
    This class provides a high-level interface for performing Grad-CAM analysis
    on YOLO models across different versions and tasks.
    
    Example:
        >>> analyzer = YoloCAMAnalyzer("model.pt", task="segmentation")
        >>> results = analyzer.analyze_performance("images/", "masks/")
        >>> analyzer.visualize_results("image.jpg", "mask.png")
    """
    
    def __init__(self, 
                 model_path: str,
                 task: str = 'auto',
                 config: Optional[YoloCAMConfig] = None):
        """Initialize the YOLO Grad-CAM analyzer.
        
        Args:
            model_path: Path to YOLO model file (.pt, .yaml, etc.)
            task: Task type ('auto', 'detection', 'segmentation', 'classification', 'pose')
            config: Optional configuration object
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model version or task is unsupported
        """
        self.logger = get_logger(__name__)
        
        # Validate inputs
        validate_model_path(model_path)
        self.model_path = model_path
        
        # Initialize configuration
        self.config = config or YoloCAMConfig()
        
        # Setup output directory
        self._setup_output_directory()
        
        # Auto-detect and initialize model
        self.model_version = self._detect_model_version()
        self.model_handler = self._create_model_handler()
        
        # Auto-detect or use specified task
        self.task_name = self._resolve_task(task)
        self.task_handler = self._create_task_handler()
        
        # Validate compatibility
        self._validate_model_task_compatibility()
        
        # Initialize CAM
        self.cam_wrapper = self._initialize_cam()
        
        self.logger.info(f"Initialized YoloCAM analyzer: {self.model_version} model for {self.task_name} task")
    
    def _setup_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
    
    def _detect_model_version(self) -> str:
        """Auto-detect YOLO model version."""
        self.logger.debug(f"Auto-detecting model version for: {self.model_path}")
        
        try:
            version = YOLOModelRegistry.auto_detect(self.model_path)
            self.logger.debug(f"Detected model version: {version}")
            return version
        except Exception as e:
            self.logger.error(f"Failed to detect model version: {e}")
            raise ValueError(f"Cannot determine YOLO model version from {self.model_path}: {e}")
    
    def _create_model_handler(self) -> BaseYOLOModel:
        """Create appropriate model handler for detected version."""
        model_class = YOLOModelRegistry.get_model(self.model_version)
        
        if model_class is None:
            available_models = YOLOModelRegistry.list_models()
            raise ValueError(
                f"Unsupported model version: {self.model_version}. "
                f"Available versions: {available_models}"
            )
        
        self.logger.debug(f"Creating model handler: {model_class.__name__}")
        return model_class(self.model_path, self.config)
    
    def _resolve_task(self, task: str) -> str:
        """Resolve task type (auto-detect if 'auto')."""
        if task == 'auto':
            self.logger.debug("Auto-detecting task type from model")
            detected_task = self.model_handler.detect_task()
            self.logger.debug(f"Detected task: {detected_task}")
            return detected_task
        
        return task.lower()
    
    def _create_task_handler(self) -> BaseTask:
        """Create appropriate task handler."""
        task_class = TaskRegistry.get_task(self.task_name)
        
        if task_class is None:
            available_tasks = TaskRegistry.list_tasks()
            raise ValueError(
                f"Unsupported task: {self.task_name}. "
                f"Available tasks: {available_tasks}"
            )
        
        self.logger.debug(f"Creating task handler: {task_class.__name__}")
        return task_class(self.config)
    
    def _validate_model_task_compatibility(self) -> None:
        """Validate that the model supports the specified task."""
        if not self.model_handler.validate_compatibility(self.task_name):
            supported_tasks = self.model_handler.supported_tasks
            raise ValueError(
                f"Model {self.model_version} does not support task '{self.task_name}'. "
                f"Supported tasks: {supported_tasks}"
            )
    
    def _initialize_cam(self) -> GradCAMWrapper:
        """Initialize CAM wrapper with appropriate settings."""
        self.logger.debug("Initializing Grad-CAM")
        
        # Get target layers based on configuration
        target_layers = self._get_target_layers()
        
        # Store target function for later use
        self.cam_target_function = self.task_handler.create_cam_target_function()
        
        return GradCAMWrapper(
            model=self.model_handler.pytorch_model,
            target_layers=target_layers,
            device=self.config.device
        )
    
    def _get_target_layers(self) -> List:
        """Get target layers for CAM based on configuration."""
        if self.config.custom_target_layers:
            # Use custom layers specified in config
            self.logger.debug("Using custom target layers from configuration")
            # This would need implementation to resolve layer names to actual layers
            # For now, fall back to component-based selection
        
        component = self.config.target_layer_component
        if component == 'auto':
            # Auto-select best component for this task
            component = self._auto_select_component()
        
        self.logger.debug(f"Getting target layers for component: {component}")
        return self.model_handler.get_target_layers(component)
    
    def _auto_select_component(self) -> str:
        """Auto-select best model component for CAM based on task."""
        # Task-specific recommendations
        task_component_map = {
            'classification': 'backbone',
            'detection': 'neck',
            'segmentation': 'neck',
            'pose': 'neck',
        }
        
        return task_component_map.get(self.task_name, 'backbone')
    
    def analyze_performance(self, 
                          image_dir: str, 
                          ground_truth_dir: str,
                          **kwargs) -> List[Dict[str, Any]]:
        """Analyze model performance on a dataset.
        
        Args:
            image_dir: Directory containing input images
            ground_truth_dir: Directory containing ground truth annotations
            **kwargs: Additional task-specific parameters
            
        Returns:
            List of performance results sorted by score (worst to best)
            
        Raises:
            FileNotFoundError: If directories don't exist
            ValueError: If no valid image pairs are found
        """
        self.logger.info(f"Starting performance analysis on {image_dir}")
        
        # Validate input directories
        validate_directory(image_dir)
        validate_directory(ground_truth_dir)
        
        # Delegate to task handler
        results = self.task_handler.analyze_performance(
            self.model_handler, image_dir, ground_truth_dir, **kwargs
        )
        
        if not results:
            raise ValueError(
                f"No valid image-ground truth pairs found in {image_dir} and {ground_truth_dir}"
            )
        
        self.logger.info(f"Analysis complete. Processed {len(results)} images")
        
        # Save results if configured
        if self.config.save_intermediate_results:
            self._save_performance_results(results)
        
        return results
    
    def generate_cam(self, 
                    image_path: str, 
                    target_class: Optional[int] = None,
                    **kwargs) -> np.ndarray:
        """Generate CAM visualization for a single image.
        
        Args:
            image_path: Path to input image
            target_class: Optional target class for CAM (task-specific)
            **kwargs: Additional CAM parameters
            
        Returns:
            CAM heatmap as numpy array
            
        Raises:
            FileNotFoundError: If image doesn't exist
        """
        self.logger.debug(f"Generating CAM for: {image_path}")
        
        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Preprocess input
        input_tensor = self.model_handler.preprocess_input(image_path)
        
        # Generate CAM with the task-specific target function
        cam_output = self.cam_wrapper.generate_cam(
            input_tensor, 
            target_function=self.cam_target_function,
            **kwargs
        )
        
        # Save raw CAM if configured
        if self.config.save_raw_cams:
            self._save_raw_cam(cam_output, image_path)
        
        return cam_output
    
    def analyze_single_image(self, 
                           image_path: str, 
                           mask_path: Optional[str] = None,
                           visualize: bool = True) -> Dict[str, Any]:
        """Analyze a single image with optional ground truth mask.
        
        Args:
            image_path: Path to input image
            mask_path: Optional path to ground truth mask
            visualize: Whether to create visualization
            
        Returns:
            Dictionary with analysis results including CAM output
        """
        self.logger.debug(f"Analyzing single image: {image_path}")
        
        # Validate image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Get model prediction
        prediction = self.task_handler._get_model_prediction(self.model_handler, image_path)
        
        # Generate CAM
        cam_output = self.generate_cam(image_path)
        
        result = {
            'image_path': image_path,
            'prediction': prediction,
            'cam_output': cam_output,
            'filename': Path(image_path).name
        }
        
        # If mask provided, compute performance metric
        if mask_path and os.path.exists(mask_path):
            ground_truth = self.task_handler.load_ground_truth(mask_path)
            score = self.task_handler.compute_performance_metric(prediction, ground_truth)
            result['ground_truth'] = ground_truth
            result['score'] = score
            result['ground_truth_path'] = mask_path
        
        # Create visualization if requested
        if visualize and self.config.save_visualizations:
            viz_paths = self.visualize_results([result], "single_image", max_images=1)
            if viz_paths:
                result['visualization_path'] = viz_paths[0]
        
        return result
    
    def visualize_results(self, 
                         image_list: List[Dict[str, Any]],
                         analysis_type: str = "analysis",
                         max_images: Optional[int] = None) -> List[str]:
        """Create comprehensive visualizations for a list of results.
        
        Args:
            image_list: List of analysis results
            analysis_type: Type description for titles ("best", "worst", etc.)
            max_images: Maximum number of images to visualize
            
        Returns:
            List of paths to saved visualization files
        """
        self.logger.info(f"Creating visualizations for {len(image_list)} {analysis_type} images")
        
        if max_images:
            image_list = image_list[:max_images]
        
        visualization_paths = []
        
        for i, result in enumerate(image_list):
            try:
                # Generate CAM for this image
                cam_output = self.generate_cam(result['image_path'])
                
                # Create visualization
                viz_image = self.task_handler.visualize_results(
                    image_path=result['image_path'],
                    prediction=result.get('prediction'),
                    ground_truth=result.get('ground_truth'),
                    cam_output=cam_output,
                    score=result.get('score'),
                    title=f"{analysis_type.title()} Example #{i+1}: {result['filename']}"
                )
                
                # Save visualization
                if self.config.save_visualizations:
                    save_path = self._save_visualization(viz_image, result['filename'], analysis_type, i)
                    visualization_paths.append(save_path)
                
            except Exception as e:
                self.logger.error(f"Failed to visualize {result['filename']}: {e}")
                continue
        
        self.logger.info(f"Created {len(visualization_paths)} visualizations")
        return visualization_paths
    
    def get_best_examples(self, 
                         results: List[Dict[str, Any]], 
                         num_examples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the best performing examples from analysis results.
        
        Args:
            results: Performance analysis results
            num_examples: Number of examples to return (uses config default if None)
            
        Returns:
            List of best examples (highest scores)
        """
        num_examples = num_examples or self.config.num_best_examples
        best_examples = results[-num_examples:]
        best_examples.reverse()  # Highest scores first
        return best_examples
    
    def get_worst_examples(self, 
                          results: List[Dict[str, Any]], 
                          num_examples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the worst performing examples from analysis results.
        
        Args:
            results: Performance analysis results
            num_examples: Number of examples to return (uses config default if None)
            
        Returns:
            List of worst examples (lowest scores)
        """
        num_examples = num_examples or self.config.num_worst_examples
        return results[:num_examples]
    
    def _save_performance_results(self, results: List[Dict[str, Any]]) -> str:
        """Save performance analysis results to file."""
        import json
        
        # Prepare results for serialization (remove non-serializable objects)
        serializable_results = []
        for result in results:
            clean_result = {
                'filename': result['filename'],
                'image_path': result['image_path'],
                'ground_truth_path': result.get('ground_truth_path'),
                'score': result['score'],
            }
            serializable_results.append(clean_result)
        
        save_path = self.output_dir / f"performance_results_{self.task_name}.json"
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.debug(f"Saved performance results to: {save_path}")
        return str(save_path)
    
    def _save_raw_cam(self, cam_output: np.ndarray, image_path: str) -> str:
        """Save raw CAM output to file."""
        import numpy as np
        
        image_name = Path(image_path).stem
        save_path = self.output_dir / f"cam_{image_name}.npy"
        np.save(save_path, cam_output)
        
        self.logger.debug(f"Saved raw CAM to: {save_path}")
        return str(save_path)
    
    def _save_visualization(self, 
                          viz_image: np.ndarray, 
                          filename: str, 
                          analysis_type: str, 
                          index: int) -> str:
        """Save visualization image to file."""
        from PIL import Image
        
        # Prepare filename
        base_name = Path(filename).stem
        save_name = f"{analysis_type}_{index+1:03d}_{base_name}.{self.config.image_save_format}"
        save_path = self.output_dir / save_name
        
        # Convert and save
        if viz_image.dtype != np.uint8:
            viz_image = (viz_image * 255).astype(np.uint8)
        
        Image.fromarray(viz_image).save(
            save_path, 
            quality=self.config.compression_quality,
            dpi=(self.config.figure_dpi, self.config.figure_dpi)
        )
        
        self.logger.debug(f"Saved visualization to: {save_path}")
        return str(save_path)
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the analyzer.
        
        Returns:
            Dictionary with analyzer configuration and capabilities
        """
        return {
            'model': self.model_handler.get_model_info(),
            'task': self.task_handler.get_task_info(),
            'cam_method': 'gradcam',
            'target_component': self.config.target_layer_component,
            'output_dir': str(self.output_dir),
            'config': self.config.to_dict(),
        }