"""Unit tests for registry system."""

import pytest
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, patch

from yolocam.core.registry import (
    YOLOModelRegistry, 
    TaskRegistry, 
    register_yolo_model, 
    register_yolo_task,
    get_registry_info,
    validate_registries
)
from yolocam.models.base_model import BaseYOLOModel
from yolocam.tasks.base_task import BaseTask


class MockYOLOModel(BaseYOLOModel):
    """Mock YOLO model for testing."""
    
    def __init__(self, model_path: str, config):
        super().__init__(model_path, config)
        self._version = 'mock_yolo'
        self._supported_tasks = ['detection', 'segmentation']
    
    def load_model(self, model_path: str):
        return Mock()
    
    def get_target_layers(self, component: str = 'backbone'):
        return [Mock()]
    
    def get_inference_model(self):
        return Mock()
    
    def preprocess_input(self, image_path: str):
        return torch.rand(1, 3, 320, 320)
    
    def detect_task(self) -> str:
        return 'detection'
    
    @property
    def supported_tasks(self):
        return self._supported_tasks
    
    @property
    def version(self) -> str:
        return self._version


class MockTask(BaseTask):
    """Mock task for testing."""
    
    def __init__(self, config):
        super().__init__(config)
        self._task_name = 'mock_task'
        self._supported_metrics = ['mock_metric']
    
    def compute_performance_metric(self, prediction, ground_truth, **kwargs):
        return 0.75
    
    def create_cam_target_function(self, **kwargs):
        return lambda x: x.sum()
    
    def load_ground_truth(self, path: str):
        return Mock()
    
    def visualize_results(self, image, prediction, ground_truth, cam_output, **kwargs):
        return Mock()
    
    @property
    def task_name(self) -> str:
        return self._task_name
    
    @property
    def supported_metrics(self):
        return self._supported_metrics
    
    @property
    def required_ground_truth_format(self) -> str:
        return 'mock_format'


class TestYOLOModelRegistry:
    """Test cases for YOLO model registry."""
    
    def test_register_model(self):
        """Test registering a model."""
        YOLOModelRegistry.register('test_yolo', MockYOLOModel, pattern=r'test.*')
        
        assert 'test_yolo' in YOLOModelRegistry._models
        assert YOLOModelRegistry._models['test_yolo'] == MockYOLOModel
        assert 'test_yolo' in YOLOModelRegistry._version_patterns
    
    def test_get_model(self):
        """Test getting a registered model."""
        YOLOModelRegistry.register('test_yolo', MockYOLOModel)
        
        model_class = YOLOModelRegistry.get_model('test_yolo')
        assert model_class == MockYOLOModel
        
        # Test non-existent model
        assert YOLOModelRegistry.get_model('nonexistent') is None
    
    def test_list_models(self):
        """Test listing registered models."""
        YOLOModelRegistry.register('yolo1', MockYOLOModel)
        YOLOModelRegistry.register('yolo2', MockYOLOModel)
        
        models = YOLOModelRegistry.list_models()
        assert 'yolo1' in models
        assert 'yolo2' in models
    
    def test_register_decorator(self):
        """Test the decorator for registering models."""
        @register_yolo_model('decorated_yolo', pattern=r'decorated.*')
        class DecoratedModel(MockYOLOModel):
            pass
        
        assert YOLOModelRegistry.get_model('decorated_yolo') == DecoratedModel
        assert 'decorated_yolo' in YOLOModelRegistry._version_patterns
    
    def test_auto_detect_from_filename(self, temp_dir):
        """Test auto-detection from filename patterns."""
        # Register models with patterns
        YOLOModelRegistry.register('yolov8', MockYOLOModel, pattern=r'.*yolo.*v?8.*')
        YOLOModelRegistry.register('yolov9', MockYOLOModel, pattern=r'.*yolo.*v?9.*')
        
        # Create test files
        yolo8_file = temp_dir / "yolov8n.pt"
        yolo9_file = temp_dir / "yolo_v9_model.pt"
        
        # Create dummy files
        yolo8_file.touch()
        yolo9_file.touch()
        
        # Mock torch.load to avoid actual loading
        with patch('torch.load'):
            assert YOLOModelRegistry.auto_detect(str(yolo8_file)) == 'yolov8'
            assert YOLOModelRegistry.auto_detect(str(yolo9_file)) == 'yolov9'
    
    def test_auto_detect_from_checkpoint(self, temp_dir):
        """Test auto-detection from checkpoint metadata."""
        YOLOModelRegistry.register('yolov8', MockYOLOModel)
        
        model_file = temp_dir / "model.pt"
        
        # Mock checkpoint with version info
        mock_checkpoint = {
            'version': 'yolov8n',
            'model': Mock()
        }
        
        with patch('torch.load', return_value=mock_checkpoint):
            version = YOLOModelRegistry.auto_detect(str(model_file))
            assert version == 'yolov8'
    
    def test_auto_detect_file_not_found(self):
        """Test auto-detection with non-existent file."""
        with pytest.raises(FileNotFoundError):
            YOLOModelRegistry.auto_detect("nonexistent.pt")
    
    def test_auto_detect_fallback(self, temp_dir):
        """Test auto-detection fallback mechanism."""
        YOLOModelRegistry.register('yolov8', MockYOLOModel)
        
        # Create file with no pattern match
        unknown_file = temp_dir / "unknown_model.pt"
        unknown_file.touch()
        
        with patch('torch.load', side_effect=Exception("Load failed")):
            # Should fall back to yolov8 as default
            version = YOLOModelRegistry.auto_detect(str(unknown_file))
            assert version == 'yolov8'
    
    def test_auto_detect_no_fallback(self, temp_dir):
        """Test auto-detection when no fallback is available."""
        # Clear registry
        YOLOModelRegistry._models.clear()
        
        unknown_file = temp_dir / "unknown.pt"
        unknown_file.touch()
        
        with patch('torch.load', side_effect=Exception("Load failed")):
            with pytest.raises(ValueError, match="Cannot detect YOLO version"):
                YOLOModelRegistry.auto_detect(str(unknown_file))


class TestTaskRegistry:
    """Test cases for task registry."""
    
    def test_register_task(self):
        """Test registering a task."""
        TaskRegistry.register('test_task', MockTask, aliases=['alias1', 'alias2'])
        
        assert 'test_task' in TaskRegistry._tasks
        assert TaskRegistry._tasks['test_task'] == MockTask
        assert TaskRegistry._task_aliases['alias1'] == 'test_task'
        assert TaskRegistry._task_aliases['alias2'] == 'test_task'
    
    def test_get_task(self):
        """Test getting a registered task."""
        TaskRegistry.register('test_task', MockTask, aliases=['alias'])
        
        # Test direct name
        task_class = TaskRegistry.get_task('test_task')
        assert task_class == MockTask
        
        # Test alias
        task_class = TaskRegistry.get_task('alias')
        assert task_class == MockTask
        
        # Test non-existent task
        assert TaskRegistry.get_task('nonexistent') is None
    
    def test_list_tasks(self):
        """Test listing registered tasks."""
        TaskRegistry.register('task1', MockTask)
        TaskRegistry.register('task2', MockTask)
        
        tasks = TaskRegistry.list_tasks()
        assert 'task1' in tasks
        assert 'task2' in tasks
    
    def test_list_aliases(self):
        """Test listing task aliases."""
        TaskRegistry.register('task1', MockTask, aliases=['alias1'])
        TaskRegistry.register('task2', MockTask, aliases=['alias2', 'alias3'])
        
        aliases = TaskRegistry.list_aliases()
        assert aliases['alias1'] == 'task1'
        assert aliases['alias2'] == 'task2'
        assert aliases['alias3'] == 'task2'
    
    def test_register_decorator(self):
        """Test the decorator for registering tasks."""
        @register_yolo_task('decorated_task', aliases=['dec_alias'])
        class DecoratedTask(MockTask):
            pass
        
        assert TaskRegistry.get_task('decorated_task') == DecoratedTask
        assert TaskRegistry.get_task('dec_alias') == DecoratedTask


class TestRegistryUtilities:
    """Test cases for registry utility functions."""
    
    def test_get_registry_info(self):
        """Test getting registry information."""
        # Register some models and tasks
        YOLOModelRegistry.register('test_model', MockYOLOModel)
        TaskRegistry.register('test_task', MockTask, aliases=['alias'])
        
        info = get_registry_info()
        
        assert 'models' in info
        assert 'tasks' in info
        assert 'test_model' in info['models']['registered']
        assert 'test_task' in info['tasks']['registered']
        assert 'alias' in info['tasks']['aliases']
        assert info['models']['count'] >= 1
        assert info['tasks']['count'] >= 1
    
    def test_validate_registries_empty(self):
        """Test registry validation with empty registries."""
        # Clear registries
        YOLOModelRegistry._models.clear()
        TaskRegistry._tasks.clear()
        
        issues = validate_registries()
        
        assert any('No YOLO models registered' in issue for issue in issues)
        assert any('detection' in issue and 'not registered' in issue for issue in issues)
        assert any('segmentation' in issue and 'not registered' in issue for issue in issues)
    
    def test_validate_registries_missing_essential_tasks(self):
        """Test registry validation with missing essential tasks."""
        # Register a model but not essential tasks
        YOLOModelRegistry.register('test_model', MockYOLOModel)
        TaskRegistry._tasks.clear()
        TaskRegistry.register('non_essential_task', MockTask)
        
        issues = validate_registries()
        
        assert any('detection' in issue and 'not registered' in issue for issue in issues)
        assert any('segmentation' in issue and 'not registered' in issue for issue in issues)
        assert not any('No YOLO models registered' in issue for issue in issues)
    
    def test_validate_registries_invalid_alias(self):
        """Test registry validation with invalid alias."""
        # Register model and tasks
        YOLOModelRegistry.register('test_model', MockYOLOModel)
        TaskRegistry.register('detection', MockTask)
        TaskRegistry.register('segmentation', MockTask)
        
        # Add invalid alias
        TaskRegistry._task_aliases['bad_alias'] = 'nonexistent_task'
        
        issues = validate_registries()
        
        assert any('bad_alias' in issue and 'unregistered task' in issue for issue in issues)
    
    def test_validate_registries_clean(self):
        """Test registry validation with clean registries."""
        # Register required components
        YOLOModelRegistry.register('test_model', MockYOLOModel)
        TaskRegistry.register('detection', MockTask)
        TaskRegistry.register('segmentation', MockTask)
        
        issues = validate_registries()
        
        assert len(issues) == 0


class TestRegistryIntegration:
    """Integration tests for registry system."""
    
    def test_model_and_task_registration_flow(self):
        """Test complete registration and lookup flow."""
        # Register model with decorator
        @register_yolo_model('flow_test_yolo', pattern=r'flow.*')
        class FlowTestModel(MockYOLOModel):
            @property
            def version(self):
                return 'flow_test_yolo'
        
        # Register task with decorator
        @register_yolo_task('flow_test_task', aliases=['flow_alias'])
        class FlowTestTask(MockTask):
            @property
            def task_name(self):
                return 'flow_test_task'
        
        # Test that everything is registered
        assert YOLOModelRegistry.get_model('flow_test_yolo') == FlowTestModel
        assert TaskRegistry.get_task('flow_test_task') == FlowTestTask
        assert TaskRegistry.get_task('flow_alias') == FlowTestTask
        
        # Test registry info includes our components
        info = get_registry_info()
        assert 'flow_test_yolo' in info['models']['registered']
        assert 'flow_test_task' in info['tasks']['registered']
        assert 'flow_alias' in info['tasks']['aliases']
        
        # Test validation passes
        issues = validate_registries()
        # Should be minimal issues (might have missing essential tasks)
        assert not any('flow_test' in issue for issue in issues)