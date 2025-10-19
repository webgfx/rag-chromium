"""
Logging configuration and utilities for the RAG system.
Provides structured logging with GPU monitoring and performance tracking.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class GPUMemoryFilter(logging.Filter):
    """Filter to add GPU memory information to log records."""
    
    def filter(self, record):
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
                record.gpu_memory_allocated = f"{memory_allocated:.2f}GB"
                record.gpu_memory_reserved = f"{memory_reserved:.2f}GB"
            else:
                record.gpu_memory_allocated = "N/A"
                record.gpu_memory_reserved = "N/A"
        except Exception:
            record.gpu_memory_allocated = "Error"
            record.gpu_memory_reserved = "Error"
        return True


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add GPU memory info if available
        if hasattr(record, 'gpu_memory_allocated'):
            log_entry['gpu_memory_allocated'] = record.gpu_memory_allocated
            log_entry['gpu_memory_reserved'] = record.gpu_memory_reserved
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
        
        return json.dumps(log_entry)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_gpu_monitoring: bool = True,
    enable_json_logging: bool = False,
    max_file_size: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_gpu_monitoring: Whether to include GPU memory info
        enable_json_logging: Whether to use JSON format for file logs
        max_file_size: Maximum size of log files before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if log directory is specified
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main log file
        log_file = log_dir / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count
        )
        
        if enable_json_logging:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Error log file
        error_file = log_dir / f"{name}_error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file, maxBytes=max_file_size, backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
    
    # Add GPU monitoring filter
    if enable_gpu_monitoring:
        gpu_filter = GPUMemoryFilter()
        for handler in logger.handlers:
            handler.addFilter(gpu_filter)
    
    return logger


class PerformanceLogger:
    """Context manager for performance logging."""
    
    def __init__(self, logger: logging.Logger, operation: str, level: str = "INFO"):
        self.logger = logger
        self.operation = operation
        self.level = getattr(logging, level.upper())
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.level, f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.log(self.level, f"Completed {self.operation} in {duration.total_seconds():.2f}s")
        else:
            self.logger.error(f"Failed {self.operation} after {duration.total_seconds():.2f}s: {exc_val}")


def log_function_call(logger: logging.Logger, level: str = "DEBUG"):
    """Decorator to log function calls with arguments and execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Log function entry
            logger.log(getattr(logging, level.upper()), f"Entering {func_name}")
            
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = datetime.now() - start_time
                logger.log(getattr(logging, level.upper()), 
                          f"Completed {func_name} in {duration.total_seconds():.3f}s")
                return result
            except Exception as e:
                duration = datetime.now() - start_time
                logger.error(f"Failed {func_name} after {duration.total_seconds():.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


def setup_monitoring_logger(
    project_name: str = "chromium-rag",
    experiment_name: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Set up a logger that integrates with Weights & Biases for experiment tracking.
    
    Args:
        project_name: W&B project name
        experiment_name: W&B experiment name
        tags: Additional tags for the experiment
    
    Returns:
        Logger configured for monitoring
    """
    logger = setup_logger(f"{project_name}.monitoring")
    
    # Try to initialize W&B
    try:
        import wandb
        
        if not wandb.run:
            wandb.init(
                project=project_name,
                name=experiment_name,
                tags=list(tags.keys()) if tags else None,
                config=tags or {}
            )
        
        class WandBHandler(logging.Handler):
            """Custom handler to send logs to Weights & Biases."""
            
            def emit(self, record):
                try:
                    log_data = {
                        'level': record.levelname,
                        'message': record.getMessage(),
                        'timestamp': record.created
                    }
                    
                    if hasattr(record, 'gpu_memory_allocated'):
                        log_data['gpu_memory'] = {
                            'allocated': record.gpu_memory_allocated,
                            'reserved': record.gpu_memory_reserved
                        }
                    
                    wandb.log(log_data)
                except Exception:
                    pass  # Don't fail if W&B logging fails
        
        wandb_handler = WandBHandler()
        wandb_handler.setLevel(logging.INFO)
        logger.addHandler(wandb_handler)
        
    except ImportError:
        logger.warning("Weights & Biases not available. Monitoring logs will only go to console/file.")
    
    return logger


# Default logger for the package
default_logger = setup_logger("rag_system")