# YoloCAM Library Design

## Core Architecture

### 1. Package Structure
```
yolocam/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── base.py              # Abstract base classes
│   ├── analyzer.py          # Main YoloCAMAnalyzer class
│   ├── registry.py          # Plugin registry system
│   └── config.py           # Configuration management
├── models/
│   ├── __init__.py
│   ├── base_model.py       # Abstract model interface
│   ├── yolov8.py          # YOLOv8 specific implementation
│   ├── yolov9.py          # YOLOv9 specific implementation
│   ├── yolov10.py         # YOLOv10 specific implementation
│   └── detector.py        # Auto-detection of model versions
├── tasks/
│   ├── __init__.py
│   ├── base_task.py       # Abstract task interface
│   ├── detection.py       # Object detection task
│   ├── segmentation.py    # Instance segmentation task
│   ├── classification.py  # Image classification task
│   └── pose.py           # Pose estimation task
├── cam/
│   ├── __init__.py
│   ├── gradcam.py         # GradCAM implementation
│   ├── eigencam.py        # EigenCAM implementation
│   └── utils.py           # CAM utilities
├── visualization/
│   ├── __init__.py
│   ├── plotter.py         # Plotting utilities
│   └── metrics.py         # Performance metrics
├── cli/
│   ├── __init__.py
│   ├── commands.py        # CLI command implementations
│   └── config.py          # CLI configuration
└── utils/
    ├── __init__.py
    ├── logging.py         # Logging configuration
    ├── io.py             # File I/O utilities
    └── validation.py     # Input validation
```

### 2. Plugin Architecture Design

#### Registry System
```python
# core/registry.py
class YOLOModelRegistry:
    """Registry for YOLO model implementations"""
    _models = {}
    
    @classmethod
    def register(cls, version: str, model_class):
        cls._models[version] = model_class
    
    @classmethod
    def get_model(cls, version: str):
        return cls._models.get(version)
    
    @classmethod
    def auto_detect(cls, model_path: str):
        # Logic to detect model version from file
        pass

# Decorator for easy registration
def register_yolo_model(version: str):
    def decorator(cls):
        YOLOModelRegistry.register(version, cls)
        return cls
    return decorator
```

#### Task Registry
```python
# core/registry.py
class TaskRegistry:
    """Registry for different YOLO tasks"""
    _tasks = {}
    
    @classmethod
    def register(cls, task_name: str, task_class):
        cls._tasks[task_name] = task_class
    
    @classmethod
    def get_task(cls, task_name: str):
        return cls._tasks.get(task_name)

def register_yolo_task(task_name: str):
    def decorator(cls):
        TaskRegistry.register(task_name, cls)
        return cls
    return decorator
```

### 3. Abstract Base Classes

#### Base Model Interface
```python
# models/base_model.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import torch.nn as nn

class BaseYOLOModel(ABC):
    """Abstract base class for YOLO model implementations"""
    
    @abstractmethod
    def load_model(self, model_path: str) -> nn.Module:
        """Load the YOLO model from path"""
        pass
    
    @abstractmethod
    def get_target_layers(self, component: str = 'backbone') -> List[nn.Module]:
        """Get recommended target layers for CAM"""
        pass
    
    @abstractmethod
    def get_inference_model(self) -> object:
        """Get model for inference operations"""
        pass
    
    @abstractmethod
    def preprocess_input(self, image_path: str) -> torch.Tensor:
        """Preprocess input image for the model"""
        pass
    
    @property
    @abstractmethod
    def supported_tasks(self) -> List[str]:
        """List of tasks supported by this model version"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Model version identifier"""
        pass
```

#### Base Task Interface
```python
# tasks/base_task.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np

class BaseTask(ABC):
    """Abstract base class for YOLO tasks"""
    
    @abstractmethod
    def compute_performance_metric(self, 
                                 prediction: Any, 
                                 ground_truth: Any) -> float:
        """Compute task-specific performance metric"""
        pass
    
    @abstractmethod
    def create_cam_target_function(self) -> callable:
        """Create CAM target function for this task"""
        pass
    
    @abstractmethod
    def visualize_results(self, 
                         image: np.ndarray,
                         prediction: Any,
                         ground_truth: Any,
                         cam_output: np.ndarray) -> np.ndarray:
        """Create task-specific visualization"""
        pass
    
    @property
    @abstractmethod
    def task_name(self) -> str:
        """Task identifier"""
        pass
```

### 4. Main Analyzer Class

```python
# core/analyzer.py
from typing import Optional, Dict, List, Any
import torch
from .registry import YOLOModelRegistry, TaskRegistry
from .config import YoloCAMConfig

class YoloCAMAnalyzer:
    """Main class for YOLO Grad-CAM analysis"""
    
    def __init__(self, 
                 model_path: str,
                 task: str = 'auto',
                 config: Optional[YoloCAMConfig] = None):
        self.config = config or YoloCAMConfig()
        
        # Auto-detect model version
        self.model_version = self._detect_model_version(model_path)
        self.model_handler = self._create_model_handler(model_path)
        
        # Auto-detect or use specified task
        self.task_name = self._detect_task(task)
        self.task_handler = self._create_task_handler()
        
        # Initialize CAM
        self.cam = self._initialize_cam()
    
    def _detect_model_version(self, model_path: str) -> str:
        """Auto-detect YOLO model version"""
        return YOLOModelRegistry.auto_detect(model_path)
    
    def _create_model_handler(self, model_path: str):
        """Create appropriate model handler"""
        model_class = YOLOModelRegistry.get_model(self.model_version)
        if not model_class:
            raise ValueError(f"Unsupported model version: {self.model_version}")
        return model_class(model_path, self.config)
    
    def _detect_task(self, task: str) -> str:
        """Detect or validate task type"""
        if task == 'auto':
            # Auto-detect from model
            return self.model_handler.detect_task()
        return task
    
    def _create_task_handler(self):
        """Create appropriate task handler"""
        task_class = TaskRegistry.get_task(self.task_name)
        if not task_class:
            raise ValueError(f"Unsupported task: {self.task_name}")
        return task_class(self.config)
    
    def analyze_performance(self, 
                          image_dir: str, 
                          ground_truth_dir: str) -> List[Dict]:
        """Analyze model performance on dataset"""
        return self.task_handler.analyze_performance(
            self.model_handler, image_dir, ground_truth_dir
        )
    
    def generate_cam(self, 
                    image_path: str, 
                    target_class: Optional[int] = None) -> np.ndarray:
        """Generate CAM visualization for image"""
        input_tensor = self.model_handler.preprocess_input(image_path)
        target_function = self.task_handler.create_cam_target_function()
        
        return self.cam(input_tensor=input_tensor, targets=[target_function])
    
    def visualize_results(self, 
                         image_path: str,
                         ground_truth_path: str,
                         save_path: Optional[str] = None) -> np.ndarray:
        """Create comprehensive visualization"""
        # Implementation for creating 1x4 visualization
        pass
```

### 5. Configuration System

```python
# core/config.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml

@dataclass
class YoloCAMConfig:
    """Configuration class for YoloCAM library"""
    
    # Model settings
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    model_input_size: int = 640
    confidence_threshold: float = 0.25
    
    # CAM settings
    cam_method: str = 'gradcam'  # 'gradcam', 'eigencam'
    target_layer_component: str = 'backbone'  # 'backbone', 'neck', 'head'
    cam_alpha: float = 0.6
    
    # Performance analysis
    performance_metrics: List[str] = field(default_factory=lambda: ['iou'])
    num_best_examples: int = 5
    num_worst_examples: int = 10
    
    # Visualization
    figure_size: tuple = (24, 6)
    colormap: str = 'jet'
    
    # Task-specific settings
    task_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'YoloCAMConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file"""
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f)
```

### 6. Example Implementation - YOLOv8 Segmentation

```python
# models/yolov8.py
@register_yolo_model('yolov8')
class YOLOv8Model(BaseYOLOModel):
    def __init__(self, model_path: str, config: YoloCAMConfig):
        self.config = config
        self.model_path = model_path
        self.pytorch_model = self.load_model(model_path)
        self.inference_model = YOLO(model_path)
    
    def load_model(self, model_path: str) -> nn.Module:
        # Implementation from current analyze.py
        pass
    
    def get_target_layers(self, component: str = 'backbone') -> List[nn.Module]:
        layer_map = {
            'backbone': [self.pytorch_model.model[9].cv2],
            'neck': [self.pytorch_model.model[15]],
            'head': [self.pytorch_model.model[22]]
        }
        return layer_map.get(component, layer_map['backbone'])
    
    @property
    def supported_tasks(self) -> List[str]:
        return ['detection', 'segmentation', 'classification']
    
    @property
    def version(self) -> str:
        return 'yolov8'

# tasks/segmentation.py
@register_yolo_task('segmentation')
class SegmentationTask(BaseTask):
    def compute_performance_metric(self, prediction, ground_truth) -> float:
        # IoU computation from current analyze.py
        pass
    
    def create_cam_target_function(self) -> callable:
        def yolo_v8_seg_target(model_output):
            return model_output[:, 4].sum()
        return yolo_v8_seg_target
    
    @property
    def task_name(self) -> str:
        return 'segmentation'
```

## Benefits of This Design

### 1. **Extensibility**
- Easy to add new YOLO versions by implementing BaseYOLOModel
- Simple task extension through BaseTask interface
- Plugin registration system for automatic discovery

### 2. **Automation-Ready**
- Configuration-driven behavior
- Consistent interfaces for testing
- Modular components for CI/CD

### 3. **Maintainability**
- Clear separation of concerns
- Abstract interfaces prevent breaking changes
- Registry pattern for loose coupling

### 4. **User-Friendly**
- Auto-detection of model versions and tasks
- Simple high-level API
- Flexible configuration system

### 5. **Performance**
- Lazy loading of components
- Efficient resource management
- Optimized for different hardware configurations

This design provides a solid foundation for building a professional, extensible YOLO Grad-CAM library that can grow with the YOLO ecosystem while maintaining backward compatibility and ease of use.