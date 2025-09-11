"""
Simple logging utility.
Nothing fancy, just gets the job done.
"""

import logging
import sys
from typing import Optional

# TODO: add log rotation
# TODO: add structured logging (JSON) for production

def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with sensible defaults.
    
    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Set level
    if level:
        logger.setLevel(getattr(logging, level.upper()))
    else:
        logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logger.level)
    
    # Create formatter
    # Simple format for development
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # TODO: add file handler for production
    # file_handler = logging.FileHandler('app.log')
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    
    return logger

# Quick test
if __name__ == "__main__":
    test_logger = get_logger(__name__)
    test_logger.info("Logger test - everything works")
    test_logger.warning("This is a warning")
    test_logger.error("This is an error")
    
    # Test different levels
    debug_logger = get_logger("debug_test", "DEBUG")
    debug_logger.debug("Debug message should appear")
    debug_logger.info("Info message")