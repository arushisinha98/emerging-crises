import time
import numpy as np
import pandas as pd

from src.data.log_utilities import _find_project_root, setup_logging, APICallLogger, DataLogger

def test_find_project_root():
    root = _find_project_root()
    assert root.exists(), "Project root should exist"
    assert (root / "src").exists(), "src directory should be present in the project root"
    assert (root / "logs").exists(), "logs directory should be created in the project root"

def test_logging_setup():
    logger = setup_logging("test_logging_setup.txt")
    logger.info("This is a test log message from the demos directory")
    logger.warning("This is a warning message to test logging levels")
    logger.error("This is an error message for testing purposes")
    log_file_path = _find_project_root() / "logs" / "test_logging_setup.txt"
    assert log_file_path.exists(), "Log file should be created in the logs directory"
    with open(log_file_path, 'r') as f:
        content = f.read()
        assert "This is a test log message from the demos directory" in content, "Log message should be present in the log file"
        assert "This is a warning message to test logging levels" in content, "Log message should be present in the log file"
        assert "This is an error message for testing purposes" in content, "Log message should be present in the log file"

def test_APICallLogger():
    logger = setup_logging("test_api_call_logger.txt")
    with APICallLogger(logger, "simulated_api_call"):
        time.sleep(0.5) # Simulate API call with delay
    log_file_path = _find_project_root() / "logs" / "test_api_call_logger.txt"
    assert log_file_path.exists(), "Log file should be created in the logs directory"
    with open(log_file_path, 'r') as f:
        content = f.read()
        assert "Starting simulated_api_call" in content, "API call start should be logged"
        assert "Completed simulated_api_call in" in content, "API call completion should be logged"

def test_DataLogger():
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'label': np.random.choice([0, 1], size=100)
    })
    logger = setup_logging("test_data_logger.txt")
    with DataLogger(logger, "simulated_data_manipulation", df_shape=sample_df.shape):
        # Simulate some data processing
        processed_df = sample_df.copy()
        processed_df['feature3'] = processed_df['feature1'] * processed_df['feature2']
        processed_df['feature4'] = processed_df['feature1'] ** 2
    log_file_path = _find_project_root() / "logs" / "test_data_logger.txt"
    assert log_file_path.exists(), "Log file should be created in the logs directory"
    with open(log_file_path, 'r') as f:
        content = f.read()
        assert "Starting simulated_data_manipulation" in content, "Data operation start should be logged"
        assert "Completed simulated_data_manipulation in" in content, "Data operation completion should be logged"