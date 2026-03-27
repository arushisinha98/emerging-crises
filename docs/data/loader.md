# Data Loader Documentation

The data loader classes in `src/data/loader.py` facilitate data acquisition from multiple financial and economic data sources. The module includes specific implementations for World Bank, OECD, Yahoo Finance, JST Macrohistory Database, and crisis labeling functionality. These loaders handle API interactions, data classification, and automated pipeline execution for financial crisis prediction datasets.

## Usage Instructions

```python
from src.data.loader import (
    WorldBankDataLoader, 
    OECDDataLoader, 
    YahooFinanceDataLoader,
    JSTDataLoader,
    CrisisLabeller
)

# World Bank data pipeline
wb_loader = WorldBankDataLoader("config.json")
wb_data = wb_loader.run_data_pipeline()

# OECD data pipeline
oecd_loader = OECDDataLoader("config.json")
oecd_data = oecd_loader.run_data_pipeline()

# Crisis labeling
labeller = CrisisLabeller("crisis_data.xlsx")
crisis_labels = labeller.create_labels(lookback_years=2)
```

## Directory Structure
```
project_root/
├── config.json             # ← Configuration file
├── data/                   # ← Downloaded datasets
│   ├── worldbank_variables.csv
│   └── oecd_variables.csv
├── logs/                   # ← Loader logs
│   ├── WorldBankDataLoader_logs.txt
│   ├── OECDDataLoader_logs.txt
│   └── CrisisLabeller_logs.txt
└── src/data/loader.py      # ← Source code
```

## Features

1. **Multi-Source Support**: Unified interface for World Bank, OECD, Yahoo Finance, and JST data sources
2. **Automatic Classification**: Topic-based variable classification using configurable rules
3. **Pipeline Automation**: End-to-end data acquisition and processing workflows
4. **API Management**: Structured API calls with logging and error handling
5. **Crisis Labeling**: Automated crisis period identification and labeling
6. **Configuration-Driven**: JSON-based configuration for flexible setup

## API Reference

### Base Classes

#### `TopicClassifier(data_tag, config_path="config.json")`

Classifier for categorizing variables into predefined buckets based on Stock & Watson (1989) methodology.

**Parameters:**
- `data_tag` (str): Identifier for the data source (e.g., 'WORLDBANK', 'OECD')
- `config_path` (str, optional): Path to JSON configuration file. Defaults to "config.json"

**Raises:**
- `ValueError`: If data_tag is not found in configuration
- `FileNotFoundError`: If configuration file is not found

**Example:**
```python
# Initialize classifier for World Bank data
classifier = TopicClassifier("WORLDBANK", "my_config.json")

# Classify a DataFrame of variables
df_with_buckets = classifier.classify_dataframe(variables_df)
```

**Key Methods:**

##### `classify_variable(row)`
Classify a single variable based on its attributes.

##### `classify_dataframe(df)`
Classify all variables in a DataFrame, adding 'bucket' and 'bucket_name' columns.

##### `add_classification_rules(bucket_id, keywords, topic_overrides=None)`
Add new classification rules for custom categorization.

#### `DataLoader(data_tag, config_path="config.json")`

Abstract base class providing common functionality for all data loaders.

**Parameters:**
- `data_tag` (str): Identifier for the data source
- `config_path` (str, optional): Path to configuration file

**Abstract Methods:**
- `fetch_indicators()`: Fetch available indicators from data source
- `download_series_data()`: Download time series data for classified variables

**Common Methods:**

##### `classify_and_save_variables(predictors, save_metadata=True)`
Classify variables and optionally save to CSV file.

##### `run_data_pipeline(save_metadata=True, upload=True)`
Execute the complete data acquisition and processing pipeline.

### Specific Data Loaders

#### `WorldBankDataLoader(config_path="config.json")`

World Bank specific data loader with API integration.

**Configuration Requirements:**
```json
{
  "WORLDBANK_TOPICS": ["1", "2", "3"],
  "WORLDBANK_SOURCES": ["World Development Indicators"],
  "DEVELOPED_MARKETS": ["USA", "GBR", "DEU"],
  "EMERGING_MARKETS": ["BRA", "CHN", "IND"]
}
```

**Example:**
```python
# Initialize World Bank loader
wb_loader = WorldBankDataLoader("config.json")

# Fetch available indicators
indicators = wb_loader.fetch_indicators()

# Download data for specific variables
data = wb_loader.download_series_data(variables=classified_vars)

# Run complete pipeline
developed_data, emerging_data = wb_loader.run_data_pipeline()
```

**Key Methods:**

##### `fetch_indicators()`
Fetch indicators from World Bank API with topic and source filtering.

**Returns:**
- `pd.DataFrame`: DataFrame containing indicator metadata

##### `download_series_data(variables=None, countries=None)`
Download time series data for classified variables.

**Parameters:**
- `variables` (pd.DataFrame, optional): Variables to download. If None, loads from metadata file
- `countries` (List[str], optional): Countries to include. If None, uses all configured markets

**Returns:**
- `pd.DataFrame`: Time series data with Country and Date columns

##### `get_series(source_id, indicator_id, country=None, start_year=1900, end_year=2025)`
Download individual time series from World Bank API.

#### `OECDDataLoader(config_path="config.json")`

OECD specific data loader for economic indicators.

**Example:**
```python
# Initialize OECD loader
oecd_loader = OECDDataLoader("config.json")

# Fetch indicators for specific countries
indicators = oecd_loader.fetch_indicators(countries=['USA', 'GBR'])

# Run pipeline
data = oecd_loader.run_data_pipeline()
```

#### `YahooFinanceDataLoader(data_tag="YahooFinance", config_path="config.json")`

Yahoo Finance data loader for financial market data.

**Example:**
```python
# Initialize Yahoo Finance loader
yf_loader = YahooFinanceDataLoader()

# Get stock symbols
symbols = yf_loader.fetch_indicators()

# Download market data
market_data = yf_loader.download_series_data(variables=symbols)
```

#### `JSTDataLoader(data_tag="JST", config_path="config.json")`

Jordà-Schularick-Taylor Macrohistory Database loader for historical financial data.

**Example:**
```python
# Initialize JST loader
jst_loader = JSTDataLoader()

# Run historical data pipeline
historical_data = jst_loader.run_data_pipeline()
```

### Crisis Labeling

#### `CrisisLabeller(filepath, sheetname="Crisis", crisis_types=None)`

Class for labeling crisis periods in financial time series data.

**Parameters:**
- `filepath` (str): Path to crisis data file (Excel format)
- `sheetname` (str, optional): Excel sheet name containing crisis data. Defaults to "Crisis"
- `crisis_types` (List[str], optional): List of crisis types to consider

**Example:**
```python
# Initialize crisis labeller
labeller = CrisisLabeller("crises.xlsx", sheetname="BankingCrises")

# Create labels with 2-year lookback
crisis_df = labeller.create_labels(
    lookback_years=2, 
    recovery_years=4,
    output_file="crisis_labels.csv"
)
```

**Key Methods:**

##### `mark_crises(df)`
Mark crisis periods in a DataFrame.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing country-date panel data

**Returns:**
- `pd.DataFrame`: DataFrame with crisis markers

##### `create_labels(lookback_years=2, recovery_years=2, output_file=None)`
Create crisis labels for machine learning with lookback and recovery periods.

**Parameters:**
- `lookback_years` (int): Years before crisis to mark as positive examples
- `recovery_years` (int): Years after crisis to exclude from training
- `output_file` (str, optional): Path to save labeled dataset

**Returns:**
- `pd.DataFrame`: DataFrame with crisis labels

## Logging

Each loader class creates logs in the `logs/` directory:

- `WorldBankDataLoader_logs.txt`: World Bank API calls and data processing
- `OECDDataLoader_logs.txt`: OECD data acquisition logs  
- `YahooFinanceDataLoader_logs.txt`: Financial market data logs
- `JSTDataLoader_logs.txt`: Historical database processing
- `TopicClassifier_logs.txt`: Variable classification operations
- `CrisisLabeller_logs.txt`: Crisis labeling operations

## Error Handling

The loaders include robust error handling:

### API Connection Errors
- Automatic retry with exponential backoff
- Graceful degradation when APIs are unavailable
- Detailed error logging with context

### Data Validation
- Column existence validation
- Data type checking
- Missing value handling

### Configuration Errors
- JSON syntax validation
- Required field verification
- Data tag existence checking

## Best Practices

### 1. Pipeline Execution
```python
# Run with proper error handling
try:
    developed_data, emerging_data = loader.run_data_pipeline(
        save_metadata=True, 
        upload=False  # Set to False for testing
    )
except Exception as e:
    logger.error(f"Pipeline failed: {e}")
```

### 2. Memory Management
```python
# For large datasets, disable metadata saving if not needed
loader.run_data_pipeline(save_metadata=False)

# Process markets separately
developed_data = loader.download_series_data(countries=loader.developed_markets)
emerging_data = loader.download_series_data(countries=loader.emerging_markets)
```

### 3. Custom Classification Rules
```python
# Add domain-specific classification rules
classifier = TopicClassifier("WORLDBANK")
classifier.add_classification_rules(
    bucket_id=7,
    keywords=["fintech", "digital.*payment", "blockchain"],
    topic_overrides={"Digital Finance": 7}
)
```

### 4. Crisis Analysis Workflow
```python
# Standard crisis prediction pipeline
labeller = CrisisLabeller("crisis_data.xlsx")

# Create labels with appropriate lookback/recovery periods
crisis_labels = labeller.create_labels(
    lookback_years=2,    # 2 years before crisis as positive examples
    recovery_years=4,    # Exclude 4 years after crisis
    output_file="labels.csv"
)
```

## Common Use Cases

### Multi-Source Data Integration
```python
# Collect data from multiple sources
wb_loader = WorldBankDataLoader()
oecd_loader = OECDDataLoader()
yf_loader = YahooFinanceDataLoader()

# Run pipelines
wb_data = wb_loader.run_data_pipeline()
oecd_data = oecd_loader.run_data_pipeline()
market_data = yf_loader.run_data_pipeline()

# Merge datasets using data_utilities
from src.data.data_utilities import merge_timeseries
combined_data = merge_timeseries([wb_data, oecd_data, market_data])
```

### Crisis Prediction Dataset Creation
```python
# Complete crisis prediction workflow
# 1. Load economic data
wb_loader = WorldBankDataLoader()
economic_data = wb_loader.run_data_pipeline()

# 2. Load financial market data  
yf_loader = YahooFinanceDataLoader()
financial_data = yf_loader.run_data_pipeline()

# 3. Merge datasets
combined_data = merge_timeseries([economic_data, financial_data])

# 4. Create crisis labels
labeller = CrisisLabeller("historical_crises.xlsx")
labeled_data = labeller.create_labels(lookback_years=2, recovery_years=3)

# 5. Upload final dataset
from src.data.data_utilities import upload_to_huggingface
upload_to_huggingface(labeled_data, "crisis-prediction-dataset", "v1.0")
```
