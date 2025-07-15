"""Unit tests for base classes and abstract interfaces."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from yolocam.models.base_model import BaseYOLOModel
from yolocam.tasks.base_task import BaseTask
from yolocam.core.config import YoloCAMConfig


class ConcreteYOLOModel(BaseYOLOModel):
    """Concrete implementation of BaseYOLOModel for testing."""
    
    def __init__(self, model_path: str, config: YoloCAMConfig):
        super().__init__(model_path, config)
        self._mock_model = Mock()
        self._mock_inference_model = Mock()
    
    def load_model(self, model_path: str):
        """Mock model loading."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return self._mock_model
    
    def get_target_layers(self, component: str = 'backbone'):
        """Mock target layer selection."""
        if component not in ['backbone', 'neck', 'head']:
            raise ValueError(f"Unsupported component: {component}")
        
        # Return different mock layers for different components
        layer_map = {
            'backbone': [Mock(name='backbone_layer')],
            'neck': [Mock(name='neck_layer')],
            'head': [Mock(name='head_layer')],
        }
        return layer_map[component]
    
    def get_inference_model(self):
        """Mock inference model."""
        return self._mock_inference_model
    
    def preprocess_input(self, image_path: str):
        """Mock input preprocessing."""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Return mock tensor with expected shape
        return torch.rand(1, 3, self.config.model_input_size, self.config.model_input_size)
    
    def detect_task(self) -> str:
        """Mock task detection."""
        return 'detection'
    
    @property
    def supported_tasks(self):
        return ['detection', 'segmentation']
    
    @property
    def version(self):
        return 'test_yolo'


class ConcreteTask(BaseTask):
    """Concrete implementation of BaseTask for testing."""
    
    def __init__(self, config: YoloCAMConfig):
        super().__init__(config)
        self._task_name = 'test_task'
    
    def compute_performance_metric(self, prediction, ground_truth, **kwargs):
        """Mock performance metric computation."""
        if prediction is None or ground_truth is None:
            return 0.0
        return 0.75  # Mock IoU score
    
    def create_cam_target_function(self, **kwargs):
        """Mock CAM target function creation."""
        def target_function(model_output):
            return model_output.sum()
        return target_function
    
    def load_ground_truth(self, path: str):
        """Mock ground truth loading."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Ground truth not found: {path}")
        
        # Return mock ground truth data
        return np.random.randint(0, 2, (240, 320), dtype=np.uint8) * 255
    
    def visualize_results(self, image, prediction, ground_truth, cam_output, **kwargs):
        """Mock visualization creation."""
        # Return mock visualization image
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @property
    def task_name(self):
        return self._task_name
    
    @property
    def supported_metrics(self):
        return ['iou', 'dice', 'accuracy']
    
    @property
    def required_ground_truth_format(self):
        return 'mask_png'


class TestBaseYOLOModel:
    """Test cases for BaseYOLOModel abstract class."""
    
    def test_concrete_implementation(self, sample_config, sample_image):
        """Test that concrete implementation works correctly."""
        model = ConcreteYOLOModel(str(sample_image), sample_config)
        
        assert model.model_path == str(sample_image)
        assert model.config == sample_config
        assert model.version == 'test_yolo'
        assert 'detection' in model.supported_tasks
    
    def test_device_detection_auto(self, sample_config, sample_image):
        """Test automatic device detection."""
        sample_config.device = 'auto'
        model = ConcreteYOLOModel(str(sample_image), sample_config)
        
        # Should detect CPU or CUDA based on availability
        assert model.device.type in ['cpu', 'cuda']
    
    def test_device_detection_explicit(self, sample_config, sample_image):
        """Test explicit device setting."""
        sample_config.device = 'cpu'
        model = ConcreteYOLOModel(str(sample_image), sample_config)
        
        assert model.device.type == 'cpu'
    
    def test_lazy_model_loading(self, sample_config, sample_image):
        """Test that model is loaded lazily."""
        model = ConcreteYOLOModel(str(sample_image), sample_config)
        
        # Model should not be loaded yet
        assert model._pytorch_model is None
        
        # Accessing pytorch_model should trigger loading
        pytorch_model = model.pytorch_model
        assert pytorch_model is not None
        assert model._pytorch_model is not None
    
    def test_lazy_inference_model_loading(self, sample_config, sample_image):
        """Test that inference model is loaded lazily."""
        model = ConcreteYOLOModel(str(sample_image), sample_config)
        
        # Inference model should not be loaded yet
        assert model._inference_model is None
        
        # Accessing inference_model should trigger loading
        inference_model = model.inference_model
        assert inference_model is not None
        assert model._inference_model is not None
    
    def test_get_target_layers_different_components(self, sample_config, sample_image):
        """Test getting target layers for different components."""
        model = ConcreteYOLOModel(str(sample_image), sample_config)
        
        backbone_layers = model.get_target_layers('backbone')
        neck_layers = model.get_target_layers('neck')
        head_layers = model.get_target_layers('head')
        
        assert len(backbone_layers) == 1
        assert len(neck_layers) == 1
        assert len(head_layers) == 1
        assert backbone_layers[0].name == 'backbone_layer'
        assert neck_layers[0].name == 'neck_layer'
        assert head_layers[0].name == 'head_layer'
    
    def test_get_target_layers_invalid_component(self, sample_config, sample_image):
        """Test getting target layers with invalid component."""
        model = ConcreteYOLOModel(str(sample_image), sample_config)
        
        with pytest.raises(ValueError, match="Unsupported component"):
            model.get_target_layers('invalid')
    
    def test_preprocess_input(self, sample_config, sample_image):
        """Test input preprocessing."""
        model = ConcreteYOLOModel(str(sample_image), sample_config)
        
        tensor = model.preprocess_input(str(sample_image))
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, sample_config.model_input_size, sample_config.model_input_size)
    
    def test_preprocess_input_file_not_found(self, sample_config, temp_dir):
        """Test preprocessing with non-existent image."""
        model = ConcreteYOLOModel("dummy_path", sample_config)
        
        with pytest.raises(FileNotFoundError):
            model.preprocess_input(str(temp_dir / "nonexistent.jpg"))
    
    def test_validate_compatibility(self, sample_config, sample_image):
        """Test task compatibility validation."""
        model = ConcreteYOLOModel(str(sample_image), sample_config)
        
        assert model.validate_compatibility('detection') is True
        assert model.validate_compatibility('segmentation') is True
        assert model.validate_compatibility('classification') is False
    
    def test_get_model_info(self, sample_config, sample_image):
        """Test getting model information."""
        model = ConcreteYOLOModel(str(sample_image), sample_config)
        
        info = model.get_model_info()
        
        assert isinstance(info, dict)
        assert info['version'] == 'test_yolo'
        assert info['supported_tasks'] == ['detection', 'segmentation']
        assert info['device'] == str(model.device)
        assert info['model_path'] == str(sample_image)
        assert info['primary_task'] == 'detection'


class TestBaseTask:
    """Test cases for BaseTask abstract class."""
    
    def test_concrete_implementation(self, sample_config):
        """Test that concrete implementation works correctly."""
        task = ConcreteTask(sample_config)
        
        assert task.config == sample_config
        assert task.task_name == 'test_task'
        assert 'iou' in task.supported_metrics
        assert task.required_ground_truth_format == 'mask_png'
    
    def test_task_config_access(self, sample_config):
        """Test access to task-specific configuration."""
        sample_config.task_configs['test_task'] = {'param': 'value'}
        task = ConcreteTask(sample_config)
        
        assert task.task_config['param'] == 'value'
    
    def test_compute_performance_metric(self, sample_config):
        """Test performance metric computation."""
        task = ConcreteTask(sample_config)
        
        # Mock prediction and ground truth
        prediction = Mock()
        ground_truth = Mock()
        
        score = task.compute_performance_metric(prediction, ground_truth)
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_compute_performance_metric_none_inputs(self, sample_config):
        """Test performance metric with None inputs."""
        task = ConcreteTask(sample_config)
        
        score = task.compute_performance_metric(None, None)
        assert score == 0.0
    
    def test_create_cam_target_function(self, sample_config):
        """Test CAM target function creation."""
        task = ConcreteTask(sample_config)
        
        target_func = task.create_cam_target_function()
        assert callable(target_func)
        
        # Test that function works with mock tensor
        mock_output = torch.rand(1, 100)
        result = target_func(mock_output)
        assert isinstance(result, torch.Tensor)
    
    def test_load_ground_truth(self, sample_config, sample_mask):
        """Test ground truth loading."""
        task = ConcreteTask(sample_config)
        
        gt_data = task.load_ground_truth(str(sample_mask))
        assert isinstance(gt_data, np.ndarray)
    
    def test_load_ground_truth_file_not_found(self, sample_config, temp_dir):
        """Test ground truth loading with non-existent file."""
        task = ConcreteTask(sample_config)
        
        with pytest.raises(FileNotFoundError):
            task.load_ground_truth(str(temp_dir / "nonexistent.png"))
    
    def test_visualize_results(self, sample_config):
        """Test result visualization."""
        task = ConcreteTask(sample_config)
        
        # Mock inputs
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        prediction = Mock()
        ground_truth = Mock()
        cam_output = np.random.rand(480, 640)
        
        viz_image = task.visualize_results(image, prediction, ground_truth, cam_output)
        
        assert isinstance(viz_image, np.ndarray)
        assert viz_image.shape == (480, 640, 3)
        assert viz_image.dtype == np.uint8
    
    def test_find_ground_truth_file(self, sample_config, sample_dataset):
        """Test finding corresponding ground truth files."""
        task = ConcreteTask(sample_config)
        
        image_file = Path(sample_dataset['images_dir']) / "image_000.jpg"
        gt_dir = sample_dataset['masks_dir']
        
        # Mock the method to test it
        gt_file = task._find_ground_truth_file(image_file, gt_dir)
        
        assert gt_file is not None
        assert gt_file.exists()
        assert gt_file.stem == image_file.stem
    
    def test_find_ground_truth_file_not_found(self, sample_config, temp_dir):
        """Test finding ground truth file when it doesn't exist."""
        task = ConcreteTask(sample_config)
        
        image_file = Path("nonexistent_image.jpg")
        gt_dir = temp_dir
        
        gt_file = task._find_ground_truth_file(image_file, gt_dir)
        assert gt_file is None
    
    def test_get_task_info(self, sample_config):
        """Test getting task information."""
        task = ConcreteTask(sample_config)
        
        info = task.get_task_info()
        
        assert isinstance(info, dict)
        assert info['task_name'] == 'test_task'
        assert info['supported_metrics'] == ['iou', 'dice', 'accuracy']
        assert info['ground_truth_format'] == 'mask_png'
        assert 'config' in info
    
    @patch('yolocam.tasks.base_task.BaseTask._get_model_prediction')
    def test_analyze_performance(self, mock_get_prediction, sample_config, sample_dataset):
        """Test performance analysis on a dataset."""
        task = ConcreteTask(sample_config)
        
        # Mock model handler
        mock_model = Mock()
        
        # Mock predictions
        mock_get_prediction.return_value = Mock()
        
        results = task.analyze_performance(
            mock_model,
            str(sample_dataset['images_dir']),
            str(sample_dataset['masks_dir'])
        )
        
        assert isinstance(results, list)
        assert len(results) == sample_dataset['num_samples']
        
        # Check result structure
        for result in results:
            assert 'filename' in result
            assert 'image_path' in result
            assert 'ground_truth_path' in result
            assert 'score' in result
            assert isinstance(result['score'], float)
    
    def test_analyze_performance_empty_dataset(self, sample_config, temp_dir):
        """Test performance analysis on empty dataset."""
        task = ConcreteTask(sample_config)
        
        # Create empty directories
        empty_images = temp_dir / "empty_images"
        empty_gt = temp_dir / "empty_gt"
        empty_images.mkdir()
        empty_gt.mkdir()
        
        mock_model = Mock()
        
        results = task.analyze_performance(
            mock_model,
            str(empty_images),
            str(empty_gt)
        )
        
        assert isinstance(results, list)
        assert len(results) == 0


class TestAbstractMethodEnforcement:
    """Test that abstract methods are properly enforced."""
    
    def test_base_yolo_model_cannot_be_instantiated(self, sample_config):
        """Test that BaseYOLOModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseYOLOModel("dummy_path", sample_config)
    
    def test_base_task_cannot_be_instantiated(self, sample_config):
        """Test that BaseTask cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTask(sample_config)
    
    def test_incomplete_yolo_model_implementation(self, sample_config):
        """Test that incomplete YOLO model implementation fails."""
        class IncompleteModel(BaseYOLOModel):
            # Missing abstract method implementations
            pass
        
        with pytest.raises(TypeError):
            IncompleteModel("dummy_path", sample_config)
    
    def test_incomplete_task_implementation(self, sample_config):
        """Test that incomplete task implementation fails."""
        class IncompleteTask(BaseTask):
            # Missing abstract method implementations
            pass
        
        with pytest.raises(TypeError):
            IncompleteTask(sample_config)