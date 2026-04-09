"""Logger Manager"""
import logging
import sys
from pathlib import Path


class Logger:
    """Unified Logger Manager"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup_logger()
        return cls._instance
    
    def _setup_logger(self):
        """Setup logger configuration"""
        self.logger = logging.getLogger('swe_bench_mm')
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    @property
    def info(self):
        return self.logger.info
    
    @property
    def error(self):
        return self.logger.error
    
    @property
    def warning(self):
        return self.logger.warning
    
    @property
    def debug(self):
        return self.logger.debug


# Global logger instance
logger = Logger()