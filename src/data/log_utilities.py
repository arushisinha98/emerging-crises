import logging
from pathlib import Path
from datetime import datetime

def _find_project_root() -> Path:
    """
    Find the project root directory by looking for key files/directories.
    Looks for src/, main.py, or other indicators starting from current directory.
    """
    current_path = Path.cwd().resolve()
    indicators = ['src', '.git']

    # Search for project root
    for path in [current_path] + list(current_path.parents):
        if any((path / indicator).exists() for indicator in indicators):
            return path
    return current_path

def setup_logging(log_file: str = "log.txt", 
                  log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up comprehensive logging for the application.
    
    Automatically creates a logs/ directory in the project root and stores
    all log files there, regardless of where the code is executed from.

    Parameters:
    -----------
    log_file: str
        Name of the log file (default: "log.txt")
    log_level: int
        Logging level (default: logging.INFO)
        
    Returns:
    --------
    logger.Logger
        Configured logger instance
    """
    # Find project root and create logs directory
    project_root = _find_project_root()
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Construct full log file path
    log_file_path = logs_dir / log_file
    
    # Create logger
    logger = logging.getLogger('data')
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter for file output
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # File handler for all logs
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Add only file handler (no console output)
    logger.addHandler(file_handler)
    
    # Log initialization message (will only go to file)
    initial_msg = f"Logging initialized. Log file: {log_file_path}"
    logger.info(initial_msg)
    
    return logger

class APICallLogger:
    """
    Context manager for logging API calls and their outcomes.
    """
    
    def __init__(self, logger: logging.Logger, operation: str, **context):
        """
        Initialize APICallLogger context manager.
        """
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        """
        Start timing and log the beginning of the operation.
        """
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation}", extra=self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Log the completion of the operation and its duration.
        """
        duration = datetime.now() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation} in {duration.total_seconds():.2f}s",
                extra=self.context)
        else:
            self.logger.error(
                f"Failed {self.operation} after {duration.total_seconds():.2f}s: {exc_val}",
                extra=self.context, exc_info=True)
        
        return False

class DataLogger:
    """
    Context manager for logging dataframe processing operations with statistics and timing.
    """

    def __init__(self, logger: logging.Logger, operation: str, df_shape=None, **context):
        """
        Initialize DataLogger context manager.
        """
        self.logger = logger
        self.operation = operation
        self.context = context
        self.df_shape = df_shape
        self.start_time = None
        self.start_missing = None
    
    def __enter__(self):
        """
        Start timing and log the beginning of the operation.
        """
        self.start_time = datetime.now()
        log_msg = f"Starting {self.operation}"
        if self.df_shape:
            log_msg += f" on DataFrame with shape {self.df_shape}"
        self.logger.info(log_msg, extra=self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Log the completion of the operation and its duration.
        """
        duration = datetime.now() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation} in {duration.total_seconds():.2f}s",
                extra=self.context)
        else:
            self.logger.error(
                f"Failed {self.operation} after {duration.total_seconds():.2f}s: {exc_val}",
                extra=self.context, exc_info=True)
        
        return False