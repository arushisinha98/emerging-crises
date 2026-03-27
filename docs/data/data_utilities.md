# Data Utilities Documentation

The data utility functions in `src/data/data_utilities.py` handle
- dataset upload to HuggingFace Hub
- time series processing
- crisis labeling.

The utilities are designed to work with pandas DataFrames and support multi-country panel data structures.

## Usage Instructions

```python
from src.data.data_utilities import (
    upload_to_huggingface,
    get_series_frequency,
    join_timelines,
    merge_timeseries,
    build_labels,
    drop_recovery
)

# Upload data to Hugging Face
upload_to_huggingface(df, "my-dataset", "my-config")

# Detect time series frequency
frequency = get_series_frequency(date_series)

# Merge multiple dataframes with time alignment
merged_df = merge_timeseries([df1, df2, df3], on='Country')

# Build crisis labels
labels = build_labels(df)
```

## Features

1. **Dataset Management**: Upload and manage datasets on Hugging Face Hub with fallback to local storage
2. **Time Series Processing**: Detect frequencies, join timelines, and merge multiple time series
3. **Crisis Analysis**: Build crisis labels for a given dataset
4. **Memory Efficiency**: Optimized algorithms for large panel datasets
5. **Error Handling**: Validation and error handling

## API Reference

### Dataset Management Functions

#### `upload_to_huggingface(data, repo_name, config_name=None)`

Upload dataset to Hugging Face Hub with fallback to local storage.

**Parameters:**
- `data` (Union[pd.DataFrame, Dict[str, pd.DataFrame]]): DataFrame or dictionary of DataFrames to upload
- `repo_name` (str): Repository name on Hugging Face Hub
- `config_name` (str, optional): Configuration/subset name for the dataset (e.g. train/test, developed/emerging)

**Raises:**
- `ValueError`: If HUGGINGFACE_TOKEN environment variable is not set
- `Exception`: If upload to Hugging Face fails (automatically falls back to local storage)

**Example:**
```python
import pandas as pd

# Single DataFrame upload
df = pd.DataFrame({'Date': ['2023-01-01', '2023-01-02'], 'Value': [100, 102]})
upload_to_huggingface(df, "financial-data", "daily-prices")

# Multiple DataFrames upload
data_dict = {
    'train': train_df,
    'test': test_df,
    'validation': val_df
}
upload_to_huggingface(data_dict, "crisis-dataset", "processed-data")
```

**Data Cleaning:**
- Automatically replaces string representations of missing values ('NA', 'N/A', '', 'nan') with `np.nan`
- Maintains data types and structure during conversion
- Note: this raises warnings for some Dataframes but is necessary for others

### Time Series Analysis Functions

#### `get_series_frequency(dates)`

Determine the frequency of a time series based on date differences.

**Parameters:**
- `dates` (pd.Series): Series containing datetime values

**Returns:**
- `str`: Frequency identifier ('D' for daily, 'M' for monthly, 'Q' for quarterly, 'Y' for yearly, or 'Unknown')

**Raises:**
- `ValueError`: If dates parameter is not a pandas Series
- Returns 'Unknown' if insufficient data is provided

**Example:**
```python
import pandas as pd

# Daily data
daily_dates = pd.Series(['2023-01-01', '2023-01-02', '2023-01-03'])
freq = get_series_frequency(daily_dates)  # Returns 'D'

# Monthly data
monthly_dates = pd.Series(['2023-01-01', '2023-02-01', '2023-03-01'])
freq = get_series_frequency(monthly_dates)  # Returns 'M'

# Quarterly data
quarterly_dates = pd.Series(['2023-01-01', '2023-04-01', '2023-07-01'])
freq = get_series_frequency(quarterly_dates)  # Returns 'Q'
```

**Frequency Detection Logic:**
- **Daily**: 1-day differences
- **Monthly**: 28-31 day differences
- **Quarterly**: 89-92 day differences  
- **Yearly**: 365-366 day differences
- **Unknown**: Any other pattern

#### `join_timelines(dates1, dates2=None)`

Create the highest frequency joint timeline from two date series.

**Parameters:**
- `dates1` (pd.Series): First timeline series
- `dates2` (pd.Series, optional): Second timeline series

**Returns:**
- `pd.DatetimeIndex`: Combined timeline with appropriate frequency

**Raises:**
- `ValueError`: If parameters are not pandas Series
- `TypeError`: If dates cannot be converted to datetime format

**Example:**
```python
import pandas as pd

# Combine monthly and quarterly data
monthly_dates = pd.Series(['2023-01-01', '2023-02-01', '2023-03-01'])
quarterly_dates = pd.Series(['2023-01-01', '2023-04-01'])

# Creates monthly timeline (higher frequency)
joint_timeline = join_timelines(monthly_dates, quarterly_dates)
```

**Frequency Priority:** Daily > Monthly > Quarterly > Yearly > Unknown

### Data Merging Functions

#### `merge_timeseries(dfs, on='Country')`

Memory-efficient merge of multiple dataframes with a common date index.

**Parameters:**
- `dfs` (List[pd.DataFrame]): List of DataFrames to merge
- `on` (str, optional): Column to merge on. Defaults to 'Country'

**Returns:**
- `pd.DataFrame`: Merged DataFrame with MultiIndex (Country, Date) or single Date index

**Raises:**
- `ValueError`: If dfs is empty or any DataFrame lacks 'Date' column

**Example:**
```python
import pandas as pd

# Economic indicators
gdp_df = pd.DataFrame({
    'Country': ['USA', 'GBR', 'USA', 'GBR'],
    'Date': ['2023-01-01', '2023-01-01', '2023-04-01', '2023-04-01'],
    'GDP': [100, 80, 102, 81]
})

# Financial indicators  
stock_df = pd.DataFrame({
    'Country': ['USA', 'GBR', 'USA', 'GBR'],
    'Date': ['2023-01-01', '2023-01-01', '2023-04-01', '2023-04-01'],
    'Stock_Index': [4000, 7000, 4100, 7200]
})

# Merge dataframes
merged_df = merge_timeseries([gdp_df, stock_df], on='Country')
print(merged_df.index.names)  # ['Country', 'Date']
```

**Features:**
- **Memory Efficient**: Processes large datasets without excessive memory usage
- **Automatic Reindexing**: Creates complete timeline grid for all countries and dates
- **Duplicate Handling**: Removes duplicate columns during merge process
- **Missing Value Handling**: Maintains proper NaN values for missing data points

### Crisis Analysis Functions

#### `build_labels(df)`

Build crisis labels for the DataFrame based on a master crisis labels dataset from Hugging Face.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing 'Country' and 'Date' columns

**Returns:**
- `np.array`: Binary array (0/1) indicating crisis periods

**Raises:**
- `ValueError`: If DataFrame lacks 'Country' or 'Date' columns, or HUGGINGFACE_USERNAME not set

**Example:**
```python
import pandas as pd

# Panel data
df = pd.DataFrame({
    'Country': ['USA', 'GBR', 'USA', 'GBR'],
    'Date': ['2008-01-01', '2008-01-01', '2009-01-01', '2009-01-01'],
    'GDP': [100, 80, 95, 78]
})

# Build crisis labels (2008 financial crisis)
crisis_labels = build_labels(df)  # Returns [1, 1, 0, 0] for 2008 crisis
```

**Requirements:**
- Environment variable `HUGGINGFACE_USERNAME` must be set
- Accesses crisis labels dataset at `{username}/crisis-labels-dataset`
- Crisis labels dataset must contain 'Country' and 'Year' columns

#### `drop_recovery(df, y, recovery_years=4)`

Remove recovery periods following crisis events to avoid look-ahead bias in modeling.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with 'Country' and 'Date' columns
- `y` (np.array): Crisis labels array
- `recovery_years` (int, optional): Number of years to drop after crisis. Defaults to 4

**Returns:**
- `Tuple[pd.DataFrame, np.array]`: Filtered DataFrame and corresponding labels

**Raises:**
- `ValueError`: If DataFrame lacks required columns

**Example:**
```python
import pandas as pd
import numpy as np

# Sample data with crisis in 2008
df = pd.DataFrame({
    'Country': ['USA'] * 6,
    'Date': ['2007-01-01', '2008-01-01', '2009-01-01', 
             '2010-01-01', '2011-01-01', '2012-01-01'],
    'GDP': [100, 95, 90, 92, 94, 96]
})

# Crisis labels (crisis in 2008)
y = np.array([0, 1, 0, 0, 0, 0])

# Drop 4-year recovery period after 2008 crisis
df_filtered, y_filtered = drop_recovery(df, y, recovery_years=4)
# Removes 2009, 2010, 2011 data (recovery period)
```

**Recovery Logic:**
- Removes data points following each crisis for specified recovery period
- Handles multiple crises per country
- Stops recovery period at next crisis if occurs within recovery window
- Preserves crisis events themselves for training

## Data Structure Requirements

### Standard Panel Data Format

Most functions expect DataFrames with this structure:

```python
df = pd.DataFrame({
    'Country': ['USA', 'GBR', 'DEU', ...],      # Country identifier
    'Date': ['2023-01-01', '2023-01-02', ...],  # Date column (any format)
    'Feature1': [1.5, 2.3, 1.8, ...],          # Numeric features
    'Feature2': [100, 110, 105, ...],          # Additional features
    # ... more columns
})
```

### Crisis Labels Dataset Format

The master crisis labels dataset should have:

```python
crisis_labels = pd.DataFrame({
    'Country': ['USA', 'GBR', 'USA', ...],     # Country names
    'Year': [2008, 2008, 1929, ...]            # Crisis years
})
```

## Best Practices

### 1. Data Preparation

```python
# Ensure consistent date formats
df['Date'] = pd.to_datetime(df['Date'])

# Standardize country names
df['Country'] = df['Country'].str.upper()

# Handle missing values explicitly
df = df.replace(['NA', 'N/A', ''], np.nan)
```

### 2. Memory Management

```python
# For large datasets, process in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    processed_chunk = merge_timeseries([chunk, other_df])
    # Process chunk
```

### 3. Crisis Analysis Workflow

```python
# Standard crisis prediction pipeline
df_merged = merge_timeseries(raw_dataframes, on='Country')
crisis_labels = build_labels(df_merged)
df_clean, labels_clean = drop_recovery(df_merged, crisis_labels)

# Now ready for machine learning
X = df_clean.drop(['Country', 'Date'], axis=1)
y = labels_clean
```

### 4. Environment Configuration

```python
# Check environment setup before operations
import os

required_vars = ['HUGGINGFACE_TOKEN', 'HUGGINGFACE_USERNAME']
for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"{var} environment variable not set")
```

## Common Use Cases

### Financial Crisis Prediction

```python
# Load multiple economic indicators
gdp_df = load_gdp_data()
stock_df = load_stock_data()
banking_df = load_banking_data()

# Merge into panel dataset
panel_df = merge_timeseries([gdp_df, stock_df, banking_df], on='Country')

# Create crisis labels and clean data
y = build_labels(panel_df)
panel_clean, y_clean = drop_recovery(panel_df, y, recovery_years=3)

# Upload to Hugging Face for sharing
upload_to_huggingface(panel_clean, "financial-crisis-data", "processed")
```

### Time Series Frequency Analysis

```python
# Analyze mixed-frequency data
monthly_data = load_monthly_indicators()
quarterly_data = load_quarterly_gdp()

# Determine frequencies
monthly_freq = get_series_frequency(monthly_data['Date'])
quarterly_freq = get_series_frequency(quarterly_data['Date'])

# Create aligned timeline
joint_timeline = join_timelines(monthly_data['Date'], quarterly_data['Date'])

# Merge with proper alignment
merged_df = merge_timeseries([monthly_data, quarterly_data])
```