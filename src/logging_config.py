"""
================================================================================
Logging Configuration Module — Structured Logging for ML Pipeline
================================================================================

This module provides a centralized logging configuration for the project.
Using proper logging instead of print statements enables:
- Log level filtering (DEBUG, INFO, WARNING, ERROR)
- File and console output
- Timestamps and source tracking
- Production-ready debugging

Usage:
    from src.logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.info("Model training started")
    logger.warning("Missing values detected")
    logger.error("Model failed to converge")
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Default log format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log directory
LOG_DIR = Path("logs")


def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_filename: Optional[str] = None,
) -> None:
    """
    Configure root logger with console and optional file handlers.
    
    Parameters
    ----------
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO).
    log_to_file : bool
        Whether to also log to a file.
    log_filename : str, optional
        Custom log filename. If None, uses timestamp.
    """
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        LOG_DIR.mkdir(exist_ok=True)
        
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"ml_pipeline_{timestamp}.log"
        
        file_handler = logging.FileHandler(LOG_DIR / log_filename)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__).
    
    Returns
    -------
    logging.Logger
        Configured logger instance.
    
    Example
    -------
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing started")
    """
    return logging.getLogger(name)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def log_model_metrics(
    logger: logging.Logger,
    model_name: str,
    metrics: dict,
    phase: str = "evaluation"
) -> None:
    """
    Log model metrics in a structured format.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    model_name : str
        Name of the model.
    metrics : dict
        Dictionary of metric names to values.
    phase : str
        Phase of training (e.g., 'training', 'validation', 'evaluation').
    """
    logger.info(f"[{phase.upper()}] {model_name} metrics:")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.info(f"  {metric_name}: {value}")


def log_data_info(
    logger: logging.Logger,
    df,
    name: str = "DataFrame"
) -> None:
    """
    Log basic DataFrame information.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    df : pd.DataFrame
        DataFrame to describe.
    name : str
        Name for the DataFrame in logs.
    """
    logger.info(f"{name} shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    missing = df.isnull().sum().sum()
    if missing > 0:
        logger.warning(f"{name} has {missing:,} missing values")
    
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"{name} has {duplicates:,} duplicate rows")


# =============================================================================
# INITIALIZATION
# =============================================================================

# Setup default logging when module is imported
setup_logging(level=logging.INFO)
