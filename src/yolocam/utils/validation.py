"""Input validation and error handling utilities."""

import os
import re
from pathlib import Path
from typing import Union, List, Optional, Any, Callable
import numpy as np
import torch
from PIL import Image

from .logging import get_logger


class YoloCAMError(Exception):
    """Base exception class for YoloCAM library."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        
        # Log the error
        logger = get_logger(__name__)
        logger.error(
            f"YoloCAMError: {message}",
            error_code=error_code,
            error_context=self.context
        )


class ModelError(YoloCAMError):
    """Exception for model-related errors."""
    pass


class ConfigurationError(YoloCAMError):
    """Exception for configuration-related errors."""
    pass


class ValidationError(YoloCAMError):
    """Exception for input validation errors."""
    pass


class DataError(YoloCAMError):
    """Exception for data-related errors."""
    pass


class CAMError(YoloCAMError):
    """Exception for CAM generation errors."""
    pass


def handle_exceptions(error_mapping: Optional[dict] = None):
    """Decorator to handle exceptions and convert them to YoloCAM exceptions.
    
    Args:
        error_mapping: Dictionary mapping exception types to YoloCAM exception types
    """
    if error_mapping is None:
        error_mapping = {
            FileNotFoundError: ValidationError,
            PermissionError: ValidationError,
            ValueError: ValidationError,
            TypeError: ValidationError,
            RuntimeError: ModelError,
            torch.cuda.OutOfMemoryError: ModelError,
        }
    
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except YoloCAMError:
                # Re-raise YoloCAM exceptions as-is
                raise
            except Exception as e:
                # Convert other exceptions to YoloCAM exceptions
                exception_type = type(e)
                yolocam_exception_type = error_mapping.get(exception_type, YoloCAMError)
                
                context = {
                    'original_exception': str(e),
                    'original_type': exception_type.__name__,
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()) if kwargs else []
                }
                
                raise yolocam_exception_type(
                    f"Error in {func.__name__}: {str(e)}",
                    error_code=f"{exception_type.__name__}_IN_{func.__name__.upper()}",
                    context=context
                ) from e
        
        return wrapper
    return decorator


@handle_exceptions()
def validate_model_path(model_path: Union[str, Path]) -> Path:
    """Validate that a model path exists and is accessible.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid or inaccessible
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise ValidationError(
            f"Model file not found: {model_path}",
            error_code="MODEL_FILE_NOT_FOUND",
            context={'path': str(model_path)}
        )
    
    if not model_path.is_file():
        raise ValidationError(
            f"Model path is not a file: {model_path}",
            error_code="MODEL_PATH_NOT_FILE",
            context={'path': str(model_path)}
        )
    
    # Check file extension
    valid_extensions = {'.pt', '.pth', '.onnx', '.yaml', '.yml'}
    if model_path.suffix.lower() not in valid_extensions:
        raise ValidationError(
            f"Unsupported model file extension: {model_path.suffix}. "
            f"Supported: {', '.join(valid_extensions)}",
            error_code="UNSUPPORTED_MODEL_EXTENSION",
            context={'path': str(model_path), 'extension': model_path.suffix}
        )
    
    # Check file permissions
    if not os.access(model_path, os.R_OK):
        raise ValidationError(
            f"Cannot read model file: {model_path}",
            error_code="MODEL_FILE_NOT_READABLE",
            context={'path': str(model_path)}
        )
    
    logger = get_logger(__name__)
    logger.debug(f"Validated model path: {model_path}")
    
    return model_path


@handle_exceptions()
def validate_image_path(image_path: Union[str, Path]) -> Path:
    """Validate that an image path exists and is a valid image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid or not a valid image
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise ValidationError(
            f"Image file not found: {image_path}",
            error_code="IMAGE_FILE_NOT_FOUND",
            context={'path': str(image_path)}
        )
    
    if not image_path.is_file():
        raise ValidationError(
            f"Image path is not a file: {image_path}",
            error_code="IMAGE_PATH_NOT_FILE",
            context={'path': str(image_path)}
        )
    
    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    if image_path.suffix.lower() not in valid_extensions:
        raise ValidationError(
            f"Unsupported image file extension: {image_path.suffix}. "
            f"Supported: {', '.join(valid_extensions)}",
            error_code="UNSUPPORTED_IMAGE_EXTENSION",
            context={'path': str(image_path), 'extension': image_path.suffix}
        )
    
    # Try to open image to validate it's actually an image
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify it's a valid image
    except Exception as e:
        raise ValidationError(
            f"Invalid image file: {image_path}. Error: {str(e)}",
            error_code="INVALID_IMAGE_FILE",
            context={'path': str(image_path), 'pil_error': str(e)}
        )
    
    # Check file permissions
    if not os.access(image_path, os.R_OK):
        raise ValidationError(
            f"Cannot read image file: {image_path}",
            error_code="IMAGE_FILE_NOT_READABLE",
            context={'path': str(image_path)}
        )
    
    logger = get_logger(__name__)
    logger.debug(f"Validated image path: {image_path}")
    
    return image_path


@handle_exceptions()
def validate_directory(directory_path: Union[str, Path], 
                      create_if_missing: bool = False,
                      check_writable: bool = False) -> Path:
    """Validate that a directory exists and is accessible.
    
    Args:
        directory_path: Path to directory
        create_if_missing: Create directory if it doesn't exist
        check_writable: Check if directory is writable
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If directory is invalid or inaccessible
    """
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        if create_if_missing:
            try:
                directory_path.mkdir(parents=True, exist_ok=True)
                logger = get_logger(__name__)
                logger.info(f"Created directory: {directory_path}")
            except Exception as e:
                raise ValidationError(
                    f"Cannot create directory: {directory_path}. Error: {str(e)}",
                    error_code="DIRECTORY_CREATION_FAILED",
                    context={'path': str(directory_path), 'error': str(e)}
                )
        else:
            raise ValidationError(
                f"Directory not found: {directory_path}",
                error_code="DIRECTORY_NOT_FOUND",
                context={'path': str(directory_path)}
            )
    
    if not directory_path.is_dir():
        raise ValidationError(
            f"Path is not a directory: {directory_path}",
            error_code="PATH_NOT_DIRECTORY",
            context={'path': str(directory_path)}
        )
    
    # Check permissions
    if not os.access(directory_path, os.R_OK):
        raise ValidationError(
            f"Cannot read directory: {directory_path}",
            error_code="DIRECTORY_NOT_READABLE",
            context={'path': str(directory_path)}
        )
    
    if check_writable and not os.access(directory_path, os.W_OK):
        raise ValidationError(
            f"Cannot write to directory: {directory_path}",
            error_code="DIRECTORY_NOT_WRITABLE",
            context={'path': str(directory_path)}
        )
    
    logger = get_logger(__name__)
    logger.debug(f"Validated directory: {directory_path}")
    
    return directory_path


@handle_exceptions()
def validate_tensor_shape(tensor: torch.Tensor, 
                         expected_shape: Optional[tuple] = None,
                         min_dims: Optional[int] = None,
                         max_dims: Optional[int] = None) -> torch.Tensor:
    """Validate tensor shape and properties.
    
    Args:
        tensor: Input tensor to validate
        expected_shape: Expected exact shape (None values are wildcards)
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
        
    Returns:
        Validated tensor
        
    Raises:
        ValidationError: If tensor doesn't meet requirements
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(
            f"Expected torch.Tensor, got {type(tensor)}",
            error_code="INVALID_TENSOR_TYPE",
            context={'actual_type': str(type(tensor))}
        )
    
    # Check dimensions
    if min_dims is not None and tensor.ndim < min_dims:
        raise ValidationError(
            f"Tensor has {tensor.ndim} dimensions, expected at least {min_dims}",
            error_code="TENSOR_INSUFFICIENT_DIMS",
            context={'actual_dims': tensor.ndim, 'min_dims': min_dims}
        )
    
    if max_dims is not None and tensor.ndim > max_dims:
        raise ValidationError(
            f"Tensor has {tensor.ndim} dimensions, expected at most {max_dims}",
            error_code="TENSOR_TOO_MANY_DIMS",
            context={'actual_dims': tensor.ndim, 'max_dims': max_dims}
        )
    
    # Check exact shape
    if expected_shape is not None:
        if len(expected_shape) != tensor.ndim:
            raise ValidationError(
                f"Tensor shape {tensor.shape} doesn't match expected shape {expected_shape}",
                error_code="TENSOR_SHAPE_MISMATCH",
                context={'actual_shape': list(tensor.shape), 'expected_shape': list(expected_shape)}
            )
        
        for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
            if expected is not None and actual != expected:
                raise ValidationError(
                    f"Tensor dimension {i} has size {actual}, expected {expected}",
                    error_code="TENSOR_DIM_SIZE_MISMATCH",
                    context={
                        'dimension': i,
                        'actual_size': actual,
                        'expected_size': expected,
                        'full_shape': list(tensor.shape)
                    }
                )
    
    # Check for NaN or infinite values
    if torch.isnan(tensor).any():
        raise ValidationError(
            "Tensor contains NaN values",
            error_code="TENSOR_CONTAINS_NAN",
            context={'shape': list(tensor.shape)}
        )
    
    if torch.isinf(tensor).any():
        raise ValidationError(
            "Tensor contains infinite values",
            error_code="TENSOR_CONTAINS_INF",
            context={'shape': list(tensor.shape)}
        )
    
    logger = get_logger(__name__)
    logger.debug(f"Validated tensor shape: {tensor.shape}")
    
    return tensor


@handle_exceptions()
def validate_array_shape(array: np.ndarray,
                        expected_shape: Optional[tuple] = None,
                        min_dims: Optional[int] = None,
                        max_dims: Optional[int] = None,
                        dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Validate numpy array shape and properties.
    
    Args:
        array: Input array to validate
        expected_shape: Expected exact shape (None values are wildcards)
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
        dtype: Expected data type
        
    Returns:
        Validated array
        
    Raises:
        ValidationError: If array doesn't meet requirements
    """
    if not isinstance(array, np.ndarray):
        raise ValidationError(
            f"Expected numpy.ndarray, got {type(array)}",
            error_code="INVALID_ARRAY_TYPE",
            context={'actual_type': str(type(array))}
        )
    
    # Check dimensions
    if min_dims is not None and array.ndim < min_dims:
        raise ValidationError(
            f"Array has {array.ndim} dimensions, expected at least {min_dims}",
            error_code="ARRAY_INSUFFICIENT_DIMS",
            context={'actual_dims': array.ndim, 'min_dims': min_dims}
        )
    
    if max_dims is not None and array.ndim > max_dims:
        raise ValidationError(
            f"Array has {array.ndim} dimensions, expected at most {max_dims}",
            error_code="ARRAY_TOO_MANY_DIMS",
            context={'actual_dims': array.ndim, 'max_dims': max_dims}
        )
    
    # Check exact shape
    if expected_shape is not None:
        if len(expected_shape) != array.ndim:
            raise ValidationError(
                f"Array shape {array.shape} doesn't match expected shape {expected_shape}",
                error_code="ARRAY_SHAPE_MISMATCH",
                context={'actual_shape': list(array.shape), 'expected_shape': list(expected_shape)}
            )
        
        for i, (actual, expected) in enumerate(zip(array.shape, expected_shape)):
            if expected is not None and actual != expected:
                raise ValidationError(
                    f"Array dimension {i} has size {actual}, expected {expected}",
                    error_code="ARRAY_DIM_SIZE_MISMATCH",
                    context={
                        'dimension': i,
                        'actual_size': actual,
                        'expected_size': expected,
                        'full_shape': list(array.shape)
                    }
                )
    
    # Check data type
    if dtype is not None and array.dtype != dtype:
        raise ValidationError(
            f"Array has dtype {array.dtype}, expected {dtype}",
            error_code="ARRAY_DTYPE_MISMATCH",
            context={'actual_dtype': str(array.dtype), 'expected_dtype': str(dtype)}
        )
    
    # Check for NaN or infinite values in float arrays
    if np.issubdtype(array.dtype, np.floating):
        if np.isnan(array).any():
            raise ValidationError(
                "Array contains NaN values",
                error_code="ARRAY_CONTAINS_NAN",
                context={'shape': list(array.shape), 'dtype': str(array.dtype)}
            )
        
        if np.isinf(array).any():
            raise ValidationError(
                "Array contains infinite values",
                error_code="ARRAY_CONTAINS_INF",
                context={'shape': list(array.shape), 'dtype': str(array.dtype)}
            )
    
    logger = get_logger(__name__)
    logger.debug(f"Validated array shape: {array.shape}, dtype: {array.dtype}")
    
    return array


def validate_config_value(value: Any, 
                         validator: Callable[[Any], bool],
                         error_message: str,
                         error_code: str) -> Any:
    """Validate a configuration value using a custom validator function.
    
    Args:
        value: Value to validate
        validator: Function that returns True if value is valid
        error_message: Error message if validation fails
        error_code: Error code for the validation failure
        
    Returns:
        Validated value
        
    Raises:
        ConfigurationError: If validation fails
    """
    try:
        if not validator(value):
            raise ConfigurationError(
                error_message,
                error_code=error_code,
                context={'value': value, 'type': str(type(value))}
            )
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        
        raise ConfigurationError(
            f"Error validating config value: {error_message}. Validator error: {str(e)}",
            error_code=f"VALIDATOR_ERROR_{error_code}",
            context={'value': value, 'validator_error': str(e)}
        )
    
    return value


def safe_operation(operation: Callable, 
                  default_return: Any = None,
                  exceptions: tuple = (Exception,),
                  log_error: bool = True) -> Any:
    """Safely execute an operation with error handling.
    
    Args:
        operation: Function to execute
        default_return: Value to return if operation fails
        exceptions: Tuple of exceptions to catch
        log_error: Whether to log the error
        
    Returns:
        Operation result or default_return if operation fails
    """
    try:
        return operation()
    except exceptions as e:
        if log_error:
            logger = get_logger(__name__)
            logger.warning(
                f"Safe operation failed: {str(e)}",
                operation_name=getattr(operation, '__name__', 'unknown'),
                error_type=type(e).__name__,
                error_message=str(e)
            )
        return default_return


class ValidationContext:
    """Context manager for validation with enhanced error reporting."""
    
    def __init__(self, operation_name: str, **context):
        self.operation_name = operation_name
        self.context = context
        self.logger = get_logger(__name__)
    
    def __enter__(self):
        self.logger.debug(f"Starting validation: {self.operation_name}", **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if issubclass(exc_type, YoloCAMError):
                # Add operation context to existing YoloCAM errors
                exc_val.context.update({
                    'validation_operation': self.operation_name,
                    **self.context
                })
            
            self.logger.error(
                f"Validation failed: {self.operation_name}",
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                **self.context
            )
        else:
            self.logger.debug(f"Validation completed: {self.operation_name}", **self.context)