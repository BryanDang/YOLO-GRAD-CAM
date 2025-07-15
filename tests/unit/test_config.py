"""Unit tests for configuration system."""

import pytest
import tempfile
from pathlib import Path
import yaml

from yolocam.core.config import YoloCAMConfig, load_default_config


class TestYoloCAMConfig:
    """Test cases for YoloCAMConfig class."""
    
    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = YoloCAMConfig()
        
        assert config.device == 'auto'
        assert config.model_input_size == 640
        assert config.cam_method == 'gradcam'
        assert config.confidence_threshold == 0.25
        assert config.num_best_examples == 5
        assert config.num_worst_examples == 10
    
    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        config = YoloCAMConfig(
            device='cuda',
            model_input_size=320,
            cam_method='eigencam',
            confidence_threshold=0.5
        )
        
        assert config.device == 'cuda'
        assert config.model_input_size == 320
        assert config.cam_method == 'eigencam'
        assert config.confidence_threshold == 0.5
    
    def test_config_validation_invalid_device(self):
        """Test validation with invalid device."""
        with pytest.raises(ValueError, match="Invalid device"):
            YoloCAMConfig(device='invalid_device')
    
    def test_config_validation_invalid_cam_method(self):
        """Test validation with invalid CAM method."""
        with pytest.raises(ValueError, match="Invalid CAM method"):
            YoloCAMConfig(cam_method='invalid_cam')
    
    def test_config_validation_invalid_alpha(self):
        """Test validation with invalid alpha value."""
        with pytest.raises(ValueError, match="cam_alpha must be between 0 and 1"):
            YoloCAMConfig(cam_alpha=1.5)
        
        with pytest.raises(ValueError, match="cam_alpha must be between 0 and 1"):
            YoloCAMConfig(cam_alpha=0.0)
    
    def test_config_validation_invalid_threshold(self):
        """Test validation with invalid threshold values."""
        with pytest.raises(ValueError, match="confidence_threshold must be between 0 and 1"):
            YoloCAMConfig(confidence_threshold=1.5)
        
        with pytest.raises(ValueError, match="iou_threshold must be between 0 and 1"):
            YoloCAMConfig(iou_threshold=-0.1)
    
    def test_config_validation_invalid_input_size(self):
        """Test validation with invalid input size."""
        with pytest.raises(ValueError, match="model_input_size must be positive"):
            YoloCAMConfig(model_input_size=0)
    
    def test_task_config_defaults(self):
        """Test that default task configurations are set up."""
        config = YoloCAMConfig()
        
        # Check that default task configs are created
        assert 'detection' in config.task_configs
        assert 'segmentation' in config.task_configs
        assert 'classification' in config.task_configs
        assert 'pose' in config.task_configs
        
        # Check segmentation config
        seg_config = config.task_configs['segmentation']
        assert 'metrics' in seg_config
        assert 'iou' in seg_config['metrics']
        assert 'target_function' in seg_config
    
    def test_task_config_merge_with_user_config(self):
        """Test merging task configs with user-provided configs."""
        custom_task_configs = {
            'segmentation': {
                'metrics': ['custom_metric'],
                'custom_param': 'value',
            }
        }
        
        config = YoloCAMConfig(task_configs=custom_task_configs)
        
        seg_config = config.task_configs['segmentation']
        assert seg_config['metrics'] == ['custom_metric']  # User value
        assert seg_config['custom_param'] == 'value'  # User value
        assert 'target_function' in seg_config  # Default value preserved
    
    def test_get_task_config(self):
        """Test getting task-specific configuration."""
        config = YoloCAMConfig()
        
        seg_config = config.get_task_config('segmentation')
        assert isinstance(seg_config, dict)
        assert 'metrics' in seg_config
        
        # Non-existent task
        empty_config = config.get_task_config('nonexistent')
        assert empty_config == {}
    
    def test_update_task_config(self):
        """Test updating task-specific configuration."""
        config = YoloCAMConfig()
        
        # Update existing task
        config.update_task_config('segmentation', {'new_param': 'value'})
        seg_config = config.get_task_config('segmentation')
        assert seg_config['new_param'] == 'value'
        
        # Create new task config
        config.update_task_config('new_task', {'param': 'value'})
        new_config = config.get_task_config('new_task')
        assert new_config['param'] == 'value'
    
    def test_config_copy(self):
        """Test creating a copy of configuration."""
        original = YoloCAMConfig(device='cuda', model_input_size=320)
        copy = original.copy()
        
        assert copy.device == original.device
        assert copy.model_input_size == original.model_input_size
        assert copy is not original
        assert copy.task_configs is not original.task_configs
    
    def test_config_merge(self):
        """Test merging configurations."""
        config1 = YoloCAMConfig(device='cpu', model_input_size=320)
        config2_dict = {
            'device': 'cuda',
            'cam_method': 'eigencam',
            'task_configs': {
                'segmentation': {'new_param': 'value'}
            }
        }
        
        merged = config1.merge(config2_dict)
        
        assert merged.device == 'cuda'  # From config2
        assert merged.model_input_size == 320  # From config1
        assert merged.cam_method == 'eigencam'  # From config2
        assert merged.task_configs['segmentation']['new_param'] == 'value'
    
    def test_config_to_dict(self):
        """Test converting configuration to dictionary."""
        config = YoloCAMConfig(device='cuda', model_input_size=320)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['device'] == 'cuda'
        assert config_dict['model_input_size'] == 320
        assert 'task_configs' in config_dict
    
    def test_config_repr(self):
        """Test string representation of configuration."""
        config = YoloCAMConfig(device='cuda', cam_method='eigencam')
        repr_str = repr(config)
        
        assert 'YoloCAMConfig' in repr_str
        assert 'device=cuda' in repr_str
        assert 'cam_method=eigencam' in repr_str


class TestConfigFileOperations:
    """Test cases for file-based configuration operations."""
    
    def test_save_and_load_config(self, temp_dir):
        """Test saving and loading configuration from file."""
        config = YoloCAMConfig(
            device='cuda',
            model_input_size=320,
            cam_method='eigencam'
        )
        
        config_path = temp_dir / "test_config.yaml"
        config.save_to_file(config_path)
        
        # Check file was created
        assert config_path.exists()
        
        # Load config back
        loaded_config = YoloCAMConfig.from_file(config_path)
        
        assert loaded_config.device == config.device
        assert loaded_config.model_input_size == config.model_input_size
        assert loaded_config.cam_method == config.cam_method
    
    def test_load_config_file_not_found(self, temp_dir):
        """Test loading configuration from non-existent file."""
        config_path = temp_dir / "nonexistent.yaml"
        
        with pytest.raises(FileNotFoundError):
            YoloCAMConfig.from_file(config_path)
    
    def test_load_config_invalid_yaml(self, temp_dir):
        """Test loading configuration from invalid YAML file."""
        config_path = temp_dir / "invalid.yaml"
        
        # Write invalid YAML
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(ValueError, match="Invalid YAML"):
            YoloCAMConfig.from_file(config_path)
    
    def test_load_config_invalid_parameters(self, temp_dir):
        """Test loading configuration with invalid parameters."""
        config_path = temp_dir / "invalid_params.yaml"
        
        # Write config with invalid parameters
        config_dict = {
            'device': 'invalid_device',
            'model_input_size': 640
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        with pytest.raises(ValueError):
            YoloCAMConfig.from_file(config_path)
    
    def test_save_config_creates_directory(self, temp_dir):
        """Test that saving config creates necessary directories."""
        config = YoloCAMConfig()
        nested_path = temp_dir / "nested" / "directory" / "config.yaml"
        
        config.save_to_file(nested_path)
        
        assert nested_path.exists()
        assert nested_path.parent.exists()


class TestConfigUtilities:
    """Test cases for configuration utility functions."""
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_default_config()
        
        assert isinstance(config, YoloCAMConfig)
        assert config.device == 'auto'
        assert config.cam_method == 'gradcam'
    
    def test_config_file_roundtrip(self, temp_dir):
        """Test that config survives save/load roundtrip without changes."""
        original = YoloCAMConfig(
            device='cuda',
            model_input_size=320,
            cam_method='eigencam',
            task_configs={
                'custom_task': {
                    'param1': 'value1',
                    'param2': [1, 2, 3],
                    'param3': {'nested': 'dict'}
                }
            }
        )
        
        config_path = temp_dir / "roundtrip.yaml"
        original.save_to_file(config_path)
        loaded = YoloCAMConfig.from_file(config_path)
        
        # Compare all important fields
        assert loaded.device == original.device
        assert loaded.model_input_size == original.model_input_size
        assert loaded.cam_method == original.cam_method
        assert loaded.task_configs == original.task_configs