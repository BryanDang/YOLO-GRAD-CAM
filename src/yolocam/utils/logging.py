"""Logging configuration and utilities for YoloCAM library."""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import traceback
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            )
        
        # Format the record
        formatted = super().format(record)
        
        return formatted


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for machine-readable logs."""
    
    def format(self, record):
        import json
        
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class YoloCAMLogger:
    """Enhanced logger with context and performance tracking."""
    
    def __init__(self, name: str, config: Optional['YoloCAMConfig'] = None):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        self.context = {}
        
        # Set up logger if not already configured
        if not self.logger.handlers:
            self._setup_logger()
    
    def _setup_logger(self):
        """Set up the logger with appropriate handlers."""
        if self.config:
            level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        else:
            level = logging.INFO
        
        self.logger.setLevel(level)
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if output directory is configured
        if self.config and hasattr(self.config, 'output_dir'):
            self._add_file_handler()
    
    def _add_file_handler(self):
        """Add file handler for persistent logging."""
        log_dir = Path(self.config.output_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler
        log_file = log_dir / 'yolocam.log'
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB files, 5 backups
        )
        
        # Use structured format for file logs
        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def set_context(self, **kwargs):
        """Set context information that will be included in all log messages."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context information."""
        self.context.clear()
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with context information."""
        extra_fields = {**self.context, **kwargs}
        
        # Create a custom LogRecord with extra fields
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=level,
            fn='',
            lno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        record.extra_fields = extra_fields
        
        self.logger.handle(record)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with full traceback."""
        import sys
        exc_info = sys.exc_info()
        
        extra_fields = {**self.context, **kwargs}
        
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=logging.ERROR,
            fn='',
            lno=0,
            msg=message,
            args=(),
            exc_info=exc_info
        )
        record.extra_fields = extra_fields
        
        self.logger.handle(record)


class PerformanceLogger:
    """Logger for performance tracking and profiling."""
    
    def __init__(self, logger: YoloCAMLogger):
        self.logger = logger
        self.timers = {}
    
    def start_timer(self, name: str):
        """Start a performance timer."""
        import time
        self.timers[name] = time.perf_counter()
        self.logger.debug(f"Started timer: {name}")
    
    def end_timer(self, name: str) -> float:
        """End a performance timer and log the duration."""
        import time
        
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        duration = time.perf_counter() - self.timers[name]
        del self.timers[name]
        
        self.logger.info(
            f"Timer '{name}' completed",
            duration_seconds=duration,
            performance_metric=True
        )
        
        return duration
    
    def log_memory_usage(self, context: str = ""):
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.logger.info(
                f"Memory usage{': ' + context if context else ''}",
                memory_rss_mb=memory_info.rss / 1024 / 1024,
                memory_vms_mb=memory_info.vms / 1024 / 1024,
                memory_metric=True
            )
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")
    
    def log_system_info(self):
        """Log system information for debugging."""
        import platform
        import torch
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            system_info.update({
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Unknown'
            })
        
        self.logger.info("System information", **system_info, system_metric=True)


# Global logger instances
_loggers: Dict[str, YoloCAMLogger] = {}
_default_config = None


def setup_logging(config: Optional['YoloCAMConfig'] = None, force: bool = False):
    """Set up global logging configuration.
    
    Args:
        config: YoloCAM configuration object
        force: Force reconfiguration even if already set up
    """
    global _default_config
    
    if _default_config is not None and not force:
        return  # Already configured
    
    _default_config = config
    
    # Configure root logger
    root_logger = logging.getLogger()
    
    if config:
        level = getattr(logging, config.log_level.upper(), logging.INFO)
    else:
        level = logging.INFO
    
    root_logger.setLevel(level)
    
    # Remove existing handlers if forcing reconfiguration
    if force:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def get_logger(name: str, config: Optional['YoloCAMConfig'] = None) -> YoloCAMLogger:
    """Get or create a YoloCAM logger instance.
    
    Args:
        name: Logger name (usually __name__)
        config: Optional configuration object
        
    Returns:
        YoloCAMLogger instance
    """
    global _loggers, _default_config
    
    # Use provided config or fall back to default
    logger_config = config or _default_config
    
    if name not in _loggers:
        _loggers[name] = YoloCAMLogger(name, logger_config)
    
    return _loggers[name]


def log_function_call(func):
    """Decorator to log function calls with arguments and return values."""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function entry
        logger.debug(
            f"Calling {func.__name__}",
            function=func.__name__,
            args_count=len(args),
            kwargs_keys=list(kwargs.keys())
        )
        
        try:
            # Call the function
            result = func(*args, **kwargs)
            
            # Log successful completion
            logger.debug(
                f"Completed {func.__name__}",
                function=func.__name__,
                success=True
            )
            
            return result
            
        except Exception as e:
            # Log exception
            logger.exception(
                f"Exception in {func.__name__}: {str(e)}",
                function=func.__name__,
                error_type=type(e).__name__
            )
            raise
    
    return wrapper


def log_performance(func):
    """Decorator to log function performance metrics."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        perf_logger = PerformanceLogger(logger)
        
        timer_name = f"{func.__module__}.{func.__name__}"
        perf_logger.start_timer(timer_name)
        
        try:
            result = func(*args, **kwargs)
            duration = perf_logger.end_timer(timer_name)
            
            # Log performance summary
            logger.info(
                f"Performance: {func.__name__}",
                function=func.__name__,
                duration_seconds=duration,
                performance_summary=True
            )
            
            return result
            
        except Exception:
            perf_logger.end_timer(timer_name)
            raise
    
    return wrapper


class LoggingContext:
    """Context manager for temporary logging context."""
    
    def __init__(self, logger: YoloCAMLogger, **context):
        self.logger = logger
        self.context = context
        self.original_context = {}
    
    def __enter__(self):
        # Save original context
        self.original_context = self.logger.context.copy()
        # Set new context
        self.logger.set_context(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original context
        self.logger.context = self.original_context
        
        # Log exception if one occurred
        if exc_type is not None:
            self.logger.exception(
                f"Exception in logging context: {exc_val}",
                exception_type=exc_type.__name__
            )