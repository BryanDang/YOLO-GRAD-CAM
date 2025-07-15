"""I/O utilities for file operations and data handling."""

import os
import json
import yaml
from pathlib import Path
from typing import Union, Dict, Any, Optional, List
import numpy as np
import cv2
from PIL import Image
import torch

from .logging import get_logger, log_performance
from .validation import (
    validate_image_path, validate_directory, handle_exceptions,
    ValidationError, DataError
)


@handle_exceptions()
@log_performance
def load_image(image_path: Union[str, Path], 
               target_size: Optional[tuple] = None,
               color_mode: str = 'RGB') -> np.ndarray:
    """Load and preprocess an image file.
    
    Args:
        image_path: Path to image file
        target_size: Optional (width, height) for resizing
        color_mode: Color mode ('RGB', 'BGR', 'GRAY')
        
    Returns:
        Image as numpy array
        
    Raises:
        ValidationError: If image path is invalid
        DataError: If image cannot be loaded or processed
    """
    logger = get_logger(__name__)
    image_path = validate_image_path(image_path)
    
    try:
        # Load image with PIL for better format support
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB' and color_mode in ['RGB', 'BGR']:
                img = img.convert('RGB')
            elif img.mode != 'L' and color_mode == 'GRAY':
                img = img.convert('L')
            
            # Resize if requested
            if target_size is not None:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            image_array = np.array(img)
            
            # Convert color space if needed
            if color_mode == 'BGR' and len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            logger.debug(
                f"Loaded image: {image_path.name}",
                shape=image_array.shape,
                dtype=str(image_array.dtype),
                color_mode=color_mode
            )
            
            return image_array
            
    except Exception as e:
        raise DataError(
            f"Failed to load image {image_path}: {str(e)}",
            error_code="IMAGE_LOAD_FAILED",
            context={'path': str(image_path), 'error': str(e)}
        )


@handle_exceptions()
@log_performance
def save_image(image: np.ndarray, 
               output_path: Union[str, Path],
               quality: int = 95,
               create_dirs: bool = True) -> Path:
    """Save an image array to file.
    
    Args:
        image: Image array to save
        output_path: Path where to save the image
        quality: JPEG quality (0-100)
        create_dirs: Create parent directories if they don't exist
        
    Returns:
        Path to saved image
        
    Raises:
        DataError: If image cannot be saved
    """
    logger = get_logger(__name__)
    output_path = Path(output_path)
    
    if create_dirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Validate image array
        if not isinstance(image, np.ndarray):
            raise DataError(
                f"Expected numpy array, got {type(image)}",
                error_code="INVALID_IMAGE_TYPE"
            )
        
        if image.ndim not in [2, 3]:
            raise DataError(
                f"Image must be 2D or 3D array, got {image.ndim}D",
                error_code="INVALID_IMAGE_DIMENSIONS",
                context={'shape': image.shape}
            )
        
        # Ensure correct data type and range
        if image.dtype != np.uint8:
            if image.dtype in [np.float32, np.float64]:
                # Assume float images are in [0, 1] range
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert to PIL Image
        if image.ndim == 2:
            pil_image = Image.fromarray(image, mode='L')
        elif image.shape[2] == 3:
            pil_image = Image.fromarray(image, mode='RGB')
        elif image.shape[2] == 4:
            pil_image = Image.fromarray(image, mode='RGBA')
        else:
            raise DataError(
                f"Unsupported number of channels: {image.shape[2]}",
                error_code="UNSUPPORTED_CHANNELS"
            )
        
        # Save with appropriate settings
        save_kwargs = {}
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            save_kwargs['quality'] = quality
            save_kwargs['optimize'] = True
        elif output_path.suffix.lower() == '.png':
            save_kwargs['optimize'] = True
        
        pil_image.save(output_path, **save_kwargs)
        
        logger.debug(
            f"Saved image: {output_path.name}",
            shape=image.shape,
            size_bytes=output_path.stat().st_size
        )
        
        return output_path
        
    except Exception as e:
        raise DataError(
            f"Failed to save image to {output_path}: {str(e)}",
            error_code="IMAGE_SAVE_FAILED",
            context={'path': str(output_path), 'error': str(e)}
        )


@handle_exceptions()
def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValidationError: If file doesn't exist
        DataError: If file cannot be parsed
    """
    logger = get_logger(__name__)
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ValidationError(
            f"Configuration file not found: {config_path}",
            error_code="CONFIG_FILE_NOT_FOUND"
        )
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise DataError(
                    f"Unsupported config file format: {config_path.suffix}",
                    error_code="UNSUPPORTED_CONFIG_FORMAT"
                )
        
        logger.debug(f"Loaded configuration from: {config_path.name}")
        return config or {}
        
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise DataError(
            f"Failed to parse configuration file {config_path}: {str(e)}",
            error_code="CONFIG_PARSE_ERROR",
            context={'path': str(config_path), 'error': str(e)}
        )


@handle_exceptions()
def save_config(config: Dict[str, Any], 
                config_path: Union[str, Path],
                create_dirs: bool = True) -> Path:
    """Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path where to save configuration
        create_dirs: Create parent directories if they don't exist
        
    Returns:
        Path to saved configuration
        
    Raises:
        DataError: If configuration cannot be saved
    """
    logger = get_logger(__name__)
    config_path = Path(config_path)
    
    if create_dirs:
        config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, 
                         sort_keys=False, allow_unicode=True)
            elif config_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise DataError(
                    f"Unsupported config file format: {config_path.suffix}",
                    error_code="UNSUPPORTED_CONFIG_FORMAT"
                )
        
        logger.debug(f"Saved configuration to: {config_path.name}")
        return config_path
        
    except Exception as e:
        raise DataError(
            f"Failed to save configuration to {config_path}: {str(e)}",
            error_code="CONFIG_SAVE_ERROR",
            context={'path': str(config_path), 'error': str(e)}
        )


@handle_exceptions()
def load_numpy_array(array_path: Union[str, Path]) -> np.ndarray:
    """Load numpy array from file.
    
    Args:
        array_path: Path to numpy array file (.npy or .npz)
        
    Returns:
        Loaded numpy array
        
    Raises:
        ValidationError: If file doesn't exist
        DataError: If file cannot be loaded
    """
    logger = get_logger(__name__)
    array_path = Path(array_path)
    
    if not array_path.exists():
        raise ValidationError(
            f"Array file not found: {array_path}",
            error_code="ARRAY_FILE_NOT_FOUND"
        )
    
    try:
        if array_path.suffix.lower() == '.npy':
            array = np.load(array_path)
        elif array_path.suffix.lower() == '.npz':
            npz_file = np.load(array_path)
            # If there's only one array, return it directly
            if len(npz_file.files) == 1:
                array = npz_file[npz_file.files[0]]
            else:
                # Return the entire npz file as a dictionary-like object
                array = npz_file
        else:
            raise DataError(
                f"Unsupported array file format: {array_path.suffix}",
                error_code="UNSUPPORTED_ARRAY_FORMAT"
            )
        
        logger.debug(
            f"Loaded array from: {array_path.name}",
            shape=array.shape if hasattr(array, 'shape') else 'multiple_arrays',
            dtype=str(array.dtype) if hasattr(array, 'dtype') else 'npz_file'
        )
        
        return array
        
    except Exception as e:
        raise DataError(
            f"Failed to load array from {array_path}: {str(e)}",
            error_code="ARRAY_LOAD_ERROR",
            context={'path': str(array_path), 'error': str(e)}
        )


@handle_exceptions()
def save_numpy_array(array: np.ndarray, 
                     array_path: Union[str, Path],
                     create_dirs: bool = True) -> Path:
    """Save numpy array to file.
    
    Args:
        array: Numpy array to save
        array_path: Path where to save array
        create_dirs: Create parent directories if they don't exist
        
    Returns:
        Path to saved array
        
    Raises:
        DataError: If array cannot be saved
    """
    logger = get_logger(__name__)
    array_path = Path(array_path)
    
    if create_dirs:
        array_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if array_path.suffix.lower() == '.npy':
            np.save(array_path, array)
        elif array_path.suffix.lower() == '.npz':
            np.savez_compressed(array_path, array=array)
        else:
            raise DataError(
                f"Unsupported array file format: {array_path.suffix}",
                error_code="UNSUPPORTED_ARRAY_FORMAT"
            )
        
        logger.debug(
            f"Saved array to: {array_path.name}",
            shape=array.shape,
            dtype=str(array.dtype),
            size_bytes=array_path.stat().st_size
        )
        
        return array_path
        
    except Exception as e:
        raise DataError(
            f"Failed to save array to {array_path}: {str(e)}",
            error_code="ARRAY_SAVE_ERROR",
            context={'path': str(array_path), 'error': str(e)}
        )


@handle_exceptions()
def find_files(directory: Union[str, Path], 
               pattern: str = "*",
               extensions: Optional[List[str]] = None,
               recursive: bool = True) -> List[Path]:
    """Find files in directory matching pattern and extensions.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern to match
        extensions: List of file extensions to include (e.g., ['.jpg', '.png'])
        recursive: Search recursively in subdirectories
        
    Returns:
        List of matching file paths
        
    Raises:
        ValidationError: If directory doesn't exist
    """
    logger = get_logger(__name__)
    directory = validate_directory(directory)
    
    try:
        if recursive:
            search_pattern = f"**/{pattern}"
            files = list(directory.glob(search_pattern))
        else:
            files = list(directory.glob(pattern))
        
        # Filter by extensions if specified
        if extensions:
            extensions = [ext.lower() for ext in extensions]
            files = [f for f in files if f.suffix.lower() in extensions]
        
        # Filter out directories
        files = [f for f in files if f.is_file()]
        
        # Sort for consistent ordering
        files.sort()
        
        logger.debug(
            f"Found {len(files)} files in {directory.name}",
            pattern=pattern,
            extensions=extensions,
            recursive=recursive
        )
        
        return files
        
    except Exception as e:
        raise DataError(
            f"Failed to find files in {directory}: {str(e)}",
            error_code="FILE_SEARCH_ERROR",
            context={'directory': str(directory), 'pattern': pattern, 'error': str(e)}
        )


@handle_exceptions()
def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get comprehensive information about a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
        
    Raises:
        ValidationError: If file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ValidationError(
            f"File not found: {file_path}",
            error_code="FILE_NOT_FOUND"
        )
    
    try:
        stat = file_path.stat()
        info = {
            'name': file_path.name,
            'stem': file_path.stem,
            'suffix': file_path.suffix,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified_time': stat.st_mtime,
            'created_time': stat.st_ctime,
            'is_file': file_path.is_file(),
            'is_dir': file_path.is_dir(),
            'absolute_path': str(file_path.absolute()),
        }
        
        # Add image-specific info if it's an image
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        if file_path.suffix.lower() in image_extensions:
            try:
                with Image.open(file_path) as img:
                    info.update({
                        'image_width': img.width,
                        'image_height': img.height,
                        'image_mode': img.mode,
                        'image_format': img.format,
                    })
            except Exception:
                pass  # Not a valid image
        
        return info
        
    except Exception as e:
        raise DataError(
            f"Failed to get file info for {file_path}: {str(e)}",
            error_code="FILE_INFO_ERROR",
            context={'path': str(file_path), 'error': str(e)}
        )


@handle_exceptions()
def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path to directory
        
    Raises:
        DataError: If directory cannot be created
    """
    return validate_directory(directory, create_if_missing=True)


def create_backup_path(original_path: Union[str, Path], 
                      backup_suffix: str = '.backup') -> Path:
    """Create a backup path for a file.
    
    Args:
        original_path: Original file path
        backup_suffix: Suffix to add for backup
        
    Returns:
        Path for backup file
    """
    original_path = Path(original_path)
    
    # Add timestamp to make backup unique
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    backup_name = f"{original_path.stem}_{timestamp}{backup_suffix}{original_path.suffix}"
    return original_path.parent / backup_name


class FileManager:
    """Context manager for safe file operations with automatic cleanup."""
    
    def __init__(self, temp_dir: Optional[Union[str, Path]] = None):
        self.temp_dir = Path(temp_dir) if temp_dir else None
        self.temp_files = []
        self.logger = get_logger(__name__)
    
    def __enter__(self):
        if self.temp_dir is None:
            import tempfile
            self.temp_dir = Path(tempfile.mkdtemp(prefix='yolocam_'))
        else:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.debug(f"Created temporary directory: {self.temp_dir}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
        
        # Clean up temporary directory
        try:
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
                self.logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up temp directory {self.temp_dir}: {e}")
    
    def create_temp_file(self, suffix: str = '', prefix: str = 'yolocam_') -> Path:
        """Create a temporary file that will be cleaned up automatically.
        
        Args:
            suffix: File suffix (e.g., '.jpg')
            prefix: File prefix
            
        Returns:
            Path to temporary file
        """
        import tempfile
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=self.temp_dir)
        os.close(fd)  # Close the file descriptor
        
        temp_path = Path(temp_path)
        self.temp_files.append(temp_path)
        
        return temp_path