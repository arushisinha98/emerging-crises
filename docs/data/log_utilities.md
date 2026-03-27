# Logging Documentation

The logging functions in `src/data/log_utilities.py` provide consistent logging format to monitor code execution. All logs are populated in the `logs/` folder, independent of where the code is executed from. Specialized Loggers, `APICallLogger` and `DataLogger` are designed to track API call and data manipulation respectively. This ensures consistent monitoring and debugging capabilities across the project workflow.

## Usage Instructions

```python
from src.data.log_utilities import setup_logging, DataLogger, APICallLogger

# Setup logging (creates logs/log.txt automatically)
logger = setup_logging("my_log_file.txt")

# Basic logging
logger.info("Starting my analysis")

# Context managers for structured operations
with APICallLogger(logger, "api_call"):
    # Your API call code  
    pass

with DataLogger(logger, "data_processing"):
    # Your data processing code
    pass
```

## Directory Structure
```
project_root/
├── logs/                   # ← All logs stored here
│   ├── demo_test.txt       # ← From demos/
│   ├── main_test.txt       # ← From main.py
│   └── log.txt             # ← Default log file
├── demos/                  # ← Can run notebooks here
├── src/                    # ← Source code
└── main.py                 # ← Can run from here
```

## Features

1. **Automatic Logs Directory Creation**: The `logs/` directory is automatically created in the project root
2. **Project Root Detection**: Finds the project root by looking for key indicators (e.g. `src/`, `.git`)
3. **Consistent Log Location**: All logs are stored in `logs/` regardless of execution location
4. **Structured Logging**: Context managers provide timing and structured information
5. **Multiple Log Levels**: Supports DEBUG, INFO, WARNING, ERROR levels
6. **File-Only Output**: Logs are written to files for clean execution

## API Reference

### Functions

#### `setup_logging(log_file: str = "log.txt", log_level: int = logging.INFO) -> logging.Logger`

Sets up logging for the application with automatic log directory creation.

**Parameters:**
- `log_file` (str, optional): Name of the log file. Defaults to "log.txt"
- `log_level` (int, optional): Logging level. Defaults to `logging.INFO`

**Returns:**
- `logging.Logger`: Configured logger instance

**Example:**
```python
# Default setup
logger = setup_logging()

# Custom log file and level
logger = setup_logging("analysis.txt", logging.DEBUG)
```

### Context Managers

#### `DataLogger(logger, operation, df_shape=None, **context)`

Context manager for logging DataFrame processing operations with timing and metadata.

**Parameters:**
- `logger` (logging.Logger): Logger instance from `setup_logging()`
- `operation` (str): Description of the operation being performed
- `df_shape` (tuple, optional): Shape of the DataFrame being processed
- `**context`: Additional context information to include in logs

**Usage:**
```python
import pandas as pd

df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

with DataLogger(logger, "data_cleaning", df_shape=df.shape, dataset="sample"):
    # Data processing operations
    df_cleaned = df.dropna()
    df_cleaned['new_col'] = df_cleaned['col1'] * 2
```

**Log Output:**
```
# File: logs/demo_test.txt
2025-07-20 13:11:37,527 - data - INFO - __enter__:117 - Starting data_cleaning on DataFrame with shape (3, 2)
2025-07-20 13:11:37,529 - data - INFO - __exit__:124 - Completed data_cleaning in 0.00s
```

#### `APICallLogger(logger, operation, **context)`

Context manager for logging API calls and their outcomes with timing information.

**Parameters:**
- `logger` (logging.Logger): Logger instance from `setup_logging()`
- `operation` (str): Description of the API operation
- `**context`: Additional context like endpoint, method, parameters

**Usage:**
```python
with APICallLogger(logger, "fetch_data", endpoint="/api/v1/data", method="GET"):
    # API call operations
    response = requests.get("https://api.example.com/data")
    data = response.json()
```

**Log Output:**
```
# File: logs/demo_test.txt
2025-07-20 13:11:42,324 - data - INFO - __enter__:84 - Starting fetch_data
2025-07-20 13:11:42,831 - data - INFO - __exit__:91 - Completed fetch_data in 0.51s
```

## Log Format

### File Logs (Detailed Format)
```
2025-07-20 13:11:30,920 - data - INFO - setup_logging:81 - Logging initialized. Log file: /path/to/logs/demo_test.txt
```

**Format Components:**
- Timestamp: `2025-07-20 13:11:30,920`
- Logger name: `data`
- Log level: `INFO`
- Function and line: `setup_logging:81`
- Message: `Logging initialized. Log file: /path/to/logs/demo_test.txt`

**Note:** All logs are written only to files. No console output is generated, ensuring clean execution without cluttering the terminal or notebook output.

## Configuration

### Log Levels

| Level | Value | Description |
|-------|-------|-------------|
| DEBUG | 10 | Detailed diagnostic info |
| INFO | 20 | General information |
| WARNING | 30 | Warning messages |
| ERROR | 40 | Error conditions |

**Note:** All log levels are written to the log file. No console output is generated.

### Project Root Detection

The logging system automatically detects the project root by searching for these indicators:

1. `src/` directory
2. `.git` directory
3. Falls back to current working directory if not found

## Best Practices

### 1. Use Descriptive Operation Names
```python
# Good
with DataLogger(logger, "clean_financial_data"):
    # processing code

# Better
with DataLogger(logger, "remove_outliers_from_stock_prices"):
    # processing code
```

### 2. Include Relevant Context
```python
with APICallLogger(logger, "fetch_stock_data", 
                   symbol="AAPL", 
                   date_range="2023-01-01_to_2023-12-31"):
    # API call code
```

### 3. Use Appropriate Log Levels
```python
logger.debug("Processing row 1000 of 10000")  # Detailed progress
logger.info("Data processing completed successfully")  # General info
logger.warning("Missing data found, using interpolation")  # Potential issues
logger.error("Failed to connect to database")  # Error conditions
```

### 4. Centralize Logger Creation
```python
# At the top of your script/notebook
logger = setup_logging("financial_analysis.txt")

# Use throughout your code
logger.info("Starting analysis...")
with DataLogger(logger, "load_data"):
    # data loading code
```