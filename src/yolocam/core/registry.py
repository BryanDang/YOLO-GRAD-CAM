"""Registry system for YOLO models and tasks."""

import re
import torch
from typing import Dict, Type, Optional, List, Any
from pathlib import Path


class YOLOModelRegistry:
    """Registry for YOLO model implementations.
    
    This registry allows dynamic registration and discovery of YOLO model
    implementations, enabling easy extension for new YOLO versions.
    """
    
    _models: Dict[str, Type['BaseYOLOModel']] = {}
    _version_patterns: Dict[str, str] = {}
    
    @classmethod
    def register(cls, version: str, model_class: Type['BaseYOLOModel'], 
                 pattern: Optional[str] = None) -> None:
        """Register a YOLO model implementation.
        
        Args:
            version: Version identifier (e.g., 'yolov8', 'yolov9')
            model_class: Model implementation class
            pattern: Optional regex pattern for auto-detection
        """
        cls._models[version] = model_class
        if pattern:
            cls._version_patterns[version] = pattern
    
    @classmethod
    def get_model(cls, version: str) -> Optional[Type['BaseYOLOModel']]:
        """Get a registered model class by version.
        
        Args:
            version: Version identifier
            
        Returns:
            Model class or None if not found
        """
        return cls._models.get(version)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """Get list of all registered model versions.
        
        Returns:
            List of version identifiers
        """
        return list(cls._models.keys())
    
    @classmethod
    def auto_detect(cls, model_path: str) -> str:
        """Auto-detect YOLO model version from file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Detected version identifier
            
        Raises:
            ValueError: If version cannot be detected
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Try to detect from filename first
        filename = model_path.name.lower()
        for version, pattern in cls._version_patterns.items():
            if re.search(pattern, filename):
                return version
        
        # Try to detect from model contents
        if model_path.suffix == '.pt':
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                version = cls._detect_from_checkpoint(checkpoint)
                if version:
                    return version
            except Exception:
                pass  # Fall back to other detection methods
        
        # Default fallback - try to determine from model architecture
        # This is a simplified heuristic - real implementation would be more sophisticated
        if 'yolo' in filename:
            if any(v in filename for v in ['v8', '8']):
                return 'yolov8'
            elif any(v in filename for v in ['v9', '9']):
                return 'yolov9'
            elif any(v in filename for v in ['v10', '10', 'v11', '11']):
                return 'yolov10'
        
        # Default to most common version
        if 'yolov8' in cls._models:
            return 'yolov8'
        
        raise ValueError(f"Cannot detect YOLO version from: {model_path}")
    
    @classmethod
    def _detect_from_checkpoint(cls, checkpoint: Dict[str, Any]) -> Optional[str]:
        """Detect version from PyTorch checkpoint metadata.
        
        Args:
            checkpoint: Loaded PyTorch checkpoint
            
        Returns:
            Detected version or None if unknown
        """
        # Check for version information in checkpoint
        if 'version' in checkpoint:
            version_str = str(checkpoint['version'])
            if 'yolov8' in version_str.lower():
                return 'yolov8'
            elif 'yolov9' in version_str.lower():
                return 'yolov9'
            elif 'yolov10' in version_str.lower():
                return 'yolov10'
        
        # Check model architecture for version hints
        if 'model' in checkpoint and hasattr(checkpoint['model'], 'yaml'):
            yaml_config = checkpoint['model'].yaml
            if isinstance(yaml_config, dict):
                # Analyze architecture characteristics
                if 'C2f' in str(yaml_config):
                    return 'yolov8'  # C2f is characteristic of YOLOv8
        
        return None


class TaskRegistry:
    """Registry for YOLO task implementations.
    
    This registry allows dynamic registration and discovery of task
    implementations for different YOLO use cases.
    """
    
    _tasks: Dict[str, Type['BaseTask']] = {}
    _task_aliases: Dict[str, str] = {}
    
    @classmethod
    def register(cls, task_name: str, task_class: Type['BaseTask'],
                 aliases: Optional[List[str]] = None) -> None:
        """Register a task implementation.
        
        Args:
            task_name: Primary task identifier
            task_class: Task implementation class
            aliases: Optional list of alternative names
        """
        cls._tasks[task_name] = task_class
        
        if aliases:
            for alias in aliases:
                cls._task_aliases[alias] = task_name
    
    @classmethod
    def get_task(cls, task_name: str) -> Optional[Type['BaseTask']]:
        """Get a registered task class by name.
        
        Args:
            task_name: Task identifier or alias
            
        Returns:
            Task class or None if not found
        """
        # Try direct lookup first
        if task_name in cls._tasks:
            return cls._tasks[task_name]
        
        # Try alias lookup
        if task_name in cls._task_aliases:
            actual_name = cls._task_aliases[task_name]
            return cls._tasks.get(actual_name)
        
        return None
    
    @classmethod
    def list_tasks(cls) -> List[str]:
        """Get list of all registered task names.
        
        Returns:
            List of primary task identifiers
        """
        return list(cls._tasks.keys())
    
    @classmethod
    def list_aliases(cls) -> Dict[str, str]:
        """Get mapping of aliases to primary task names.
        
        Returns:
            Dictionary mapping aliases to primary names
        """
        return cls._task_aliases.copy()


# Decorator functions for easy registration

def register_yolo_model(version: str, pattern: Optional[str] = None):
    """Decorator to register a YOLO model implementation.
    
    Args:
        version: Version identifier
        pattern: Optional regex pattern for auto-detection
        
    Example:
        @register_yolo_model('yolov8', r'yolo.*v?8')
        class YOLOv8Model(BaseYOLOModel):
            pass
    """
    def decorator(cls):
        YOLOModelRegistry.register(version, cls, pattern)
        return cls
    return decorator


def register_yolo_task(task_name: str, aliases: Optional[List[str]] = None):
    """Decorator to register a YOLO task implementation.
    
    Args:
        task_name: Primary task identifier
        aliases: Optional list of alternative names
        
    Example:
        @register_yolo_task('segmentation', ['seg', 'instance_seg'])
        class SegmentationTask(BaseTask):
            pass
    """
    def decorator(cls):
        TaskRegistry.register(task_name, cls, aliases)
        return cls
    return decorator


# Registry inspection utilities

def get_registry_info() -> Dict[str, Any]:
    """Get comprehensive information about registered components.
    
    Returns:
        Dictionary with registry status and capabilities
    """
    return {
        'models': {
            'registered': YOLOModelRegistry.list_models(),
            'count': len(YOLOModelRegistry._models),
        },
        'tasks': {
            'registered': TaskRegistry.list_tasks(),
            'aliases': TaskRegistry.list_aliases(),
            'count': len(TaskRegistry._tasks),
        },
    }


def validate_registries() -> List[str]:
    """Validate registry consistency and return any issues.
    
    Returns:
        List of validation error messages (empty if all OK)
    """
    issues = []
    
    # Check that we have at least one model registered
    if not YOLOModelRegistry._models:
        issues.append("No YOLO models registered")
    
    # Check that we have at least basic tasks registered
    essential_tasks = ['detection', 'segmentation']
    registered_tasks = TaskRegistry.list_tasks()
    
    for task in essential_tasks:
        if task not in registered_tasks:
            issues.append(f"Essential task '{task}' not registered")
    
    # Check for circular aliases
    aliases = TaskRegistry._task_aliases
    for alias, target in aliases.items():
        if target not in TaskRegistry._tasks:
            issues.append(f"Alias '{alias}' points to unregistered task '{target}'")
    
    return issues