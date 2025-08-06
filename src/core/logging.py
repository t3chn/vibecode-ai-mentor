"""Logging configuration for VibeCode AI Mentor."""

import logging
import sys
from typing import Optional

from src.core.config import settings


def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        name: Logger name, defaults to root logger
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Format based on environment
        if settings.is_development:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        else:
            formatter = logging.Formatter(
                '{"time": "%(asctime)s", "name": "%(name)s", '
                '"level": "%(levelname)s", "message": "%(message)s"}'
            )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(settings.log_level)
    
    return logger


# Create default logger
logger = setup_logging("vibecode")