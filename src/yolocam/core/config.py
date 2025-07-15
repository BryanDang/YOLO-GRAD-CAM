"""Configuration system for YoloCAM library."""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


@dataclass
class YoloCAMConfig:
    """Configuration class for YoloCAM library.
    
    This class centralizes all configuration options for the library,
    supporting both programmatic configuration and file-based configuration.
    """
    
    # Model settings
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'
    model_input_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300
    
    # CAM settings
    target_layer_component: str = 'backbone'  # 'backbone', 'neck', 'head', 'auto'
    custom_target_layers: Optional[List[str]] = None
    cam_alpha: float = 0.6  # Overlay transparency
    use_guided_gradcam: bool = False
    
    # Performance analysis settings
    performance_metrics: List[str] = field(default_factory=lambda: ['iou'])
    num_best_examples: int = 5
    num_worst_examples: int = 10
    analysis_batch_size: int = 1
    save_intermediate_results: bool = False
    
    # Visualization settings
    figure_size: tuple = (24, 6)
    figure_dpi: int = 100
    colormap: str = 'jet'
    show_confidence_scores: bool = True
    show_class_labels: bool = True
    font_size: int = 12
    
    # I/O settings
    output_dir: str = './results'
    save_visualizations: bool = True
    save_raw_cams: bool = False
    image_save_format: str = 'png'  # 'png', 'jpg', 'pdf'
    compression_quality: int = 95  # For JPEG
    
    # Logging and debugging
    log_level: str = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    verbose: bool = True
    progress_bar: bool = True
    profiling_enabled: bool = False
    
    # Task-specific configurations
    task_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Advanced settings
    memory_limit_gb: Optional[float] = None
    num_workers: int = 0  # For data loading
    pin_memory: bool = True
    benchmark_mode: bool = False  # PyTorch benchmark mode
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_config()
        self._setup_defaults()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Device validation
        valid_devices = ['auto', 'cpu', 'cuda', 'mps']
        if self.device not in valid_devices:
            raise ValueError(f"Invalid device: {self.device}. Must be one of {valid_devices}")
        
        
        # Target layer component validation
        valid_components = ['backbone', 'neck', 'head', 'auto']
        if self.target_layer_component not in valid_components:
            raise ValueError(f"Invalid target layer component: {self.target_layer_component}. Must be one of {valid_components}")
        
        # Range validations
        if not 0 < self.cam_alpha <= 1:
            raise ValueError("cam_alpha must be between 0 and 1")
        
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if not 0 <= self.iou_threshold <= 1:
            raise ValueError("iou_threshold must be between 0 and 1")
        
        if self.model_input_size <= 0:
            raise ValueError("model_input_size must be positive")
        
        # File format validation
        valid_formats = ['png', 'jpg', 'jpeg', 'pdf']
        if self.image_save_format.lower() not in valid_formats:
            raise ValueError(f"Invalid image format: {self.image_save_format}. Must be one of {valid_formats}")
    
    def _setup_defaults(self):
        """Set up default task configurations if not provided."""
        default_task_configs = {
            'detection': {
                'metrics': ['mAP', 'precision', 'recall'],
                'target_function': 'max_confidence',
            },
            'segmentation': {
                'metrics': ['iou', 'dice', 'pixel_accuracy'],
                'target_function': 'segmentation_score',
                'mask_threshold': 0.5,
            },
            'classification': {
                'metrics': ['accuracy', 'top5_accuracy'],
                'target_function': 'class_score',
            },
            'pose': {
                'metrics': ['oks', 'pck'],
                'target_function': 'keypoint_confidence',
                'keypoint_threshold': 0.3,
            },
        }
        
        # Merge with user-provided configs, keeping user values
        for task, default_config in default_task_configs.items():
            if task not in self.task_configs:
                self.task_configs[task] = default_config.copy()
            else:
                # Merge defaults with user config, preferring user values
                merged_config = default_config.copy()
                merged_config.update(self.task_configs[task])
                self.task_configs[task] = merged_config
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'YoloCAMConfig':
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            YoloCAMConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file format is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f) or {}
            
            return cls(**config_dict)
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {config_path}: {e}")
        except TypeError as e:
            raise ValueError(f"Invalid configuration parameters: {e}")
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file.
        
        Args:
            config_path: Path where to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary and handle non-serializable types
        config_dict = asdict(self)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, 
                     sort_keys=False, allow_unicode=True)
    
    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        """Get configuration for a specific task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Task-specific configuration dictionary
        """
        return self.task_configs.get(task_name, {})
    
    def update_task_config(self, task_name: str, config: Dict[str, Any]) -> None:
        """Update configuration for a specific task.
        
        Args:
            task_name: Name of the task
            config: Configuration dictionary to merge
        """
        if task_name not in self.task_configs:
            self.task_configs[task_name] = {}
        
        self.task_configs[task_name].update(config)
    
    def copy(self) -> 'YoloCAMConfig':
        """Create a deep copy of the configuration.
        
        Returns:
            New YoloCAMConfig instance with same values
        """
        config_dict = asdict(self)
        return YoloCAMConfig(**config_dict)
    
    def merge(self, other: Union['YoloCAMConfig', Dict[str, Any]]) -> 'YoloCAMConfig':
        """Merge with another configuration.
        
        Args:
            other: Another config object or dictionary
            
        Returns:
            New merged configuration
        """
        current_dict = asdict(self)
        
        if isinstance(other, YoloCAMConfig):
            other_dict = asdict(other)
        else:
            other_dict = other
        
        # Deep merge dictionaries
        merged_dict = self._deep_merge(current_dict, other_dict)
        return YoloCAMConfig(**merged_dict)
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries.
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary (takes precedence)
            
        Returns:
            Merged dictionary
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return asdict(self)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"YoloCAMConfig(device={self.device}, model_input_size={self.model_input_size})"


def load_default_config() -> YoloCAMConfig:
    """Load default configuration.
    
    Returns:
        Default YoloCAMConfig instance
    """
    return YoloCAMConfig()


def load_config_from_env() -> YoloCAMConfig:
    """Load configuration from environment variables.
    
    Environment variables should be prefixed with 'YOLOCAM_'.
    For example: YOLOCAM_DEVICE=cuda, YOLOCAM_CAM_METHOD=eigencam
    
    Returns:
        YoloCAMConfig instance with values from environment
    """
    config_dict = {}
    prefix = 'YOLOCAM_'
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Convert environment variable name to config field name
            field_name = key[len(prefix):].lower()
            
            # Handle nested configurations (e.g., YOLOCAM_TASK_CONFIGS_DETECTION_METRICS)
            if 'task_configs' in field_name:
                # This would need more sophisticated parsing for nested configs
                continue
            
            # Type conversion based on default config
            default_config = YoloCAMConfig()
            if hasattr(default_config, field_name):
                default_value = getattr(default_config, field_name)
                
                # Convert string value to appropriate type
                if isinstance(default_value, bool):
                    config_dict[field_name] = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(default_value, int):
                    config_dict[field_name] = int(value)
                elif isinstance(default_value, float):
                    config_dict[field_name] = float(value)
                elif isinstance(default_value, list):
                    config_dict[field_name] = value.split(',')
                else:
                    config_dict[field_name] = value
    
    return YoloCAMConfig(**config_dict)