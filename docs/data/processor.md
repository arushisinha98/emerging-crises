# Data Preprocessing Documentation

The preprocessing pipeline in `src/data/processor.py` provides data cleaning and imputation methods for financial time series datasets. The `PreprocessPipeline` class handles missing value imputation, data resampling, outlier removal, and feature engineering while maintaining temporal integrity and preventing look-ahead bias. The pipeline supports both single-country and multi-country panel datasets.

## Usage Instructions

```python
from src.data.processor import PreprocessPipeline

# Initialize pipeline with data from Hugging Face
pipeline = PreprocessPipeline("worldbank-data", subset="developed")

# Initialize with custom DataFrame
pipeline = PreprocessPipeline("custom-analysis", df=my_dataframe)

# Basic preprocessing chain
pipeline = (PreprocessPipeline("financial-data")
    .trim_timeseries(completeness=0.4)
    .drop_columns(completeness=0.6)
    .drop_static()
    .resample(frequency='M')
    .forward_fill(n=3)
    .knn_iterative_imputer(k=5))

# View results
pipeline.print_info()
```

## Features

1. **Temporal Integrity**: All operations respect chronological order and prevent look-ahead bias
2. **Multiple Imputation Methods**: Forward fill, backward fill, ARIMA, and KNN-based imputation
3. **Data Quality Control**: Automated column/row dropping based on completeness thresholds
4. **Flexible Resampling**: Support for daily, monthly, quarterly, and yearly aggregation
5. **Detailed Logging**: Tracking of all preprocessing operations with statistics

## API Reference

### Main Class

#### `PreprocessPipeline(data_tag, subset=None, split='train', df=None)`

Main preprocessing pipeline class for financial time series data.

**Parameters:**
- `data_tag` (str): Identifier for the pipeline or data source name
- `subset` (str, optional): Subset of countries to process ("developed", "emerging")
- `split` (str, optional): Dataset split to use. Defaults to 'train'
- `df` (pd.DataFrame, optional): Custom DataFrame to process. If None, loads from Hugging Face

**Example:**
```python
# Load from Hugging Face Hub
pipeline = PreprocessPipeline("crisis-dataset", subset="developed")

# Use custom data
pipeline = PreprocessPipeline("my-analysis", df=custom_df)

# Access the processed DataFrame
processed_data = pipeline.df
```

### Data Quality Methods

#### `trim_timeseries(completeness=0.4)`

Trim the DataFrame to include only dates where column completeness exceeds threshold.

**Parameters:**
- `completeness` (float, optional): Minimum completeness threshold (0.0-1.0). Defaults to 0.4

**Returns:**
- `PreprocessPipeline`: Self for method chaining

**Example:**
```python
# Keep only dates where at least 60% of columns have data
pipeline.trim_timeseries(completeness=0.6)
```

#### `drop_columns(columns=[], completeness=0.6)`

Drop specified columns and/or columns below completeness threshold.

**Parameters:**
- `columns` (List[str], optional): Specific columns to drop. Defaults to empty list
- `completeness` (float, optional): Minimum completeness threshold. Defaults to 0.6

**Returns:**
- `PreprocessPipeline`: Self for method chaining

**Example:**
```python
# Drop specific columns and those with <80% completeness
pipeline.drop_columns(['old_indicator', 'deprecated_var'], completeness=0.8)
```

#### `drop_static()`

Remove columns and rows with little to no variation (≤2 unique values).

**Returns:**
- `PreprocessPipeline`: Self for method chaining

**Example:**
```python
# Remove constant or near-constant variables
pipeline.drop_static()
```

### Time Series Operations

#### `resample(frequency='M', mapping=None)`

Aggregate DataFrame to specified frequency with custom aggregation functions.

**Parameters:**
- `frequency` (str, optional): Target frequency ('D', 'M', 'Q', 'Y'). Defaults to 'M'
- `mapping` (Dict[str, List[str]], optional): Column-to-aggregation mappings

**Returns:**
- `PreprocessPipeline`: Self for method chaining

**Available Aggregations:** 'sum', 'last', 'min', 'max', 'mean', 'median'

**Example:**
```python
# Monthly aggregation with custom rules
mapping = {
    'sum': ['trade_volume', 'transactions'],
    'mean': ['prices', 'returns'],
    'last': ['policy_rate', 'exchange_rate']
}
pipeline.resample(frequency='M', mapping=mapping)
```

### Missing Value Imputation

#### `forward_fill(n=None, latest_only=True, columns=None)`

Fill missing values using forward fill method with look-ahead bias prevention.

**Parameters:**
- `n` (int, optional): Maximum number of periods to fill. If None, fills all
- `latest_only` (bool, optional): If True, only fills latest missing values. Defaults to True
- `columns` (List[str], optional): Columns to process. If None, processes all numeric columns

**Returns:**
- `PreprocessPipeline`: Self for method chaining

**Example:**
```python
# Forward fill up to 3 periods for specific indicators
pipeline.forward_fill(n=3, columns=['gdp', 'unemployment'])

# Fill all missing values in latest periods only
pipeline.forward_fill(latest_only=True)
```

#### `zero_fill()`

Fill missing values with zero for series that were never reported.

**Returns:**
- `PreprocessPipeline`: Self for method chaining

**Example:**
```python
# Replace missing values with zeros
pipeline.zero_fill()
```

#### `backfill(fill='mean', columns=None)`

Fill missing values using statistical measures or zero.

**Parameters:**
- `fill` (str, optional): Fill method ('mean', 'median', 'mode', 'zero'). Defaults to 'mean'
- `columns` (List[str], optional): Columns to process. If None, processes all

**Returns:**
- `PreprocessPipeline`: Self for method chaining

**Example:**
```python
# Backfill with country-specific means
pipeline.backfill(fill='mean')

# Backfill specific columns with median
pipeline.backfill(fill='median', columns=['inflation', 'growth'])
```

#### `knn_iterative_imputer(columns=None, k=5, mask=None, feature_columns=None, distance_metric='euclidean', normalize_features=True)`

KNN-based imputation without look-ahead bias using temporal constraints.

**Parameters:**
- `columns` (List[str], optional): Columns to impute. If None, imputes all numeric columns
- `k` (int, optional): Number of nearest neighbors. Defaults to 5
- `mask` (pd.DataFrame, optional): Imputation mask. If None, imputes all missing values
- `feature_columns` (List[str], optional): Columns for distance computation
- `distance_metric` (str, optional): Distance metric ('euclidean', 'manhattan'). Defaults to 'euclidean'
- `normalize_features` (bool, optional): Whether to normalize features. Defaults to True

**Returns:**
- `PreprocessPipeline`: Self for method chaining

**Example:**
```python
# KNN imputation with 7 neighbors
pipeline.knn_iterative_imputer(k=7, normalize_features=True)

# Impute specific columns using Manhattan distance
pipeline.knn_iterative_imputer(
    columns=['gdp', 'inflation'],
    k=3,
    distance_metric='manhattan'
)
```

#### `arima_fill(columns=None)`

Fill missing values using ARIMA time series models with automatic order selection.

**Parameters:**
- `columns` (List[str], optional): Columns to process. If None, processes all numeric columns

**Returns:**
- `PreprocessPipeline`: Self for method chaining

**Example:**
```python
# ARIMA imputation for all columns
pipeline.arima_fill()

# ARIMA imputation for specific economic indicators
pipeline.arima_fill(columns=['gdp_growth', 'unemployment_rate'])
```

### Utility Methods

#### `get_imputation_mask(columns=None, include_before=False, include_after=False)`

Generate mask indicating which values need imputation.

**Parameters:**
- `columns` (List[str], optional): Columns to check. If None, checks all columns
- `include_before` (bool, optional): Include values before first observation. Defaults to False
- `include_after` (bool, optional): Include values after last observation. Defaults to False

**Returns:**
- `pd.DataFrame`: Boolean mask indicating missing values

**Example:**
```python
# Create mask for specific columns
mask = pipeline.get_imputation_mask(['gdp', 'inflation'])

# Use mask with KNN imputation
pipeline.knn_iterative_imputer(mask=mask)
```

#### `print_info(countries=None)`

Display data quality information in markdown table format.

**Parameters:**
- `countries` (List[str], optional): Countries to analyze. If None, analyzes all countries

**Example:**
```python
# Print overall data quality
pipeline.print_info()

# Print quality for specific countries
pipeline.print_info(['United States', 'Germany', 'Japan'])
```

**Output Format:**
```
| Dataset | Countries | Avg Temporal Range | Avg Completeness |
|---------|-----------|-------------------|------------------|
| crisis-data | 25 | 1990-2023 | 78.5% |
```

## Pipeline Logging

The preprocessing pipeline maintains detailed logs of all operations:

### Log File Location
```
logs/PreprocessPipeline_logs.txt
```

### Log Content Example
```
2025-08-28 14:15:30,920 - data - INFO - __init__:45 - PreprocessPipeline initialized for crisis-dataset
2025-08-28 14:15:31,124 - data - INFO - trim_timeseries:89 - Trimmed DataFrame to dates from 1990-01-01
2025-08-28 14:15:31,340 - data - INFO - drop_columns:125 - Dropped 15 columns with completeness < 0.6
2025-08-28 14:15:31,567 - data - INFO - drop_static:145 - 8 static columns removed, 12 static rows removed
2025-08-28 14:15:32,890 - data - INFO - knn_iterative_imputer:285 - KNN imputation completed. Imputed 1,234 values using k=5 neighbors.
```

### Operation Tracking

Each preprocessing operation is logged in `preprocess_log` attribute:

```python
# View preprocessing history
print(pipeline.preprocess_log)
# Output: ['trim_timeseries(0.4)', 'drop_columns(columns=[], completeness=0.6)', 'drop_static()', ...]
```

## Method Chaining Patterns

### Basic Cleaning Pipeline
```python
pipeline = (PreprocessPipeline("financial-data")
    .trim_timeseries(completeness=0.5)
    .drop_columns(completeness=0.7)
    .drop_static()
    .resample(frequency='Q'))
```

### Advanced Imputation Pipeline
```python
pipeline = (PreprocessPipeline("macro-indicators")
    .drop_columns(completeness=0.8)
    .forward_fill(n=2, latest_only=True)
    .backfill(fill='mean')
    .knn_iterative_imputer(k=7, normalize_features=True)
    .arima_fill())
```

### Custom Frequency Pipeline
```python
# Monthly aggregation with specific rules
mapping = {
    'last': ['policy_rate', 'unemployment'],
    'mean': ['stock_prices', 'exchange_rates'],
    'sum': ['trade_volume']
}

pipeline = (PreprocessPipeline("high-freq-data")
    .resample(frequency='M', mapping=mapping)
    .forward_fill(n=1)
    .zero_fill())
```

## Data Requirements

### Expected DataFrame Structure

The pipeline expects DataFrames with this structure:

```python
df = pd.DataFrame({
    'Country': ['USA', 'GBR', 'DEU', ...],      # Country identifier (optional)
    'Date': ['2023-01-01', '2023-01-02', ...],  # Date column (required)
    'Indicator1': [1.5, 2.3, np.nan, ...],     # Numeric indicators
    'Indicator2': [100, np.nan, 105, ...],     # More indicators
    # ... additional columns
})
```

### Multi-Index Support

For panel data with MultiIndex:

```python
df = df.set_index(['Country', 'Date'])
# Pipeline automatically handles MultiIndex structures
```

## Error Handling

The pipeline includes robust error handling:

### Missing Required Columns
- Validates presence of 'Date' column
- Handles absence of 'Country' column gracefully
- Provides informative error messages

### Data Type Issues
- Automatic datetime conversion for 'Date' column
- Numeric type validation for indicator columns
- Graceful handling of mixed data types

### Temporal Constraints
- Ensures chronological ordering in all operations
- Prevents look-ahead bias in imputation methods
- Validates date ranges and frequencies

## Best Practices

### 1. Pipeline Design
```python
# Build pipelines incrementally
pipeline = PreprocessPipeline("dataset")
pipeline.print_info()  # Check initial state

pipeline.trim_timeseries(0.5)
pipeline.print_info()  # Check after trimming

# Continue building...
```

### 2. Completeness Thresholds
```python
# Use progressive thresholds
pipeline = (PreprocessPipeline("sparse-data")
    .drop_columns(completeness=0.8)    # Strict for column removal
    .trim_timeseries(completeness=0.4)  # Lenient for time range
    .forward_fill(n=2))                # Conservative gap filling
```

### 3. Imputation Strategy
```python
# Layer imputation methods by data quality
pipeline = (PreprocessPipeline("mixed-quality-data")
    .forward_fill(n=1, latest_only=True)      # Fill recent gaps
    .backfill(fill='mean')                    # Handle systematic missing
    .knn_iterative_imputer(k=5)               # Fill remaining gaps
    .zero_fill())                             # Final safety net
```

### 4. Memory Management
```python
# For large datasets, process by country groups
countries_group1 = ['USA', 'GBR', 'DEU', 'FRA']
countries_group2 = ['JPN', 'CAN', 'AUS', 'CHE']

for countries in [countries_group1, countries_group2]:
    subset_pipeline = PreprocessPipeline("large-dataset", df=df[df['Country'].isin(countries)])
    # Process subset...
```

## Common Use Cases

### Financial Crisis Prediction
```python
# Prepare crisis prediction dataset
pipeline = (PreprocessPipeline("crisis-indicators")
    .trim_timeseries(completeness=0.6)
    .drop_columns(completeness=0.8)
    .resample(frequency='Q')
    .forward_fill(n=1)
    .knn_iterative_imputer(k=5))

# Prepare for modeling
X = pipeline.df.drop(['Country', 'Date'], axis=1)
```

### High-Frequency to Low-Frequency Conversion
```python
# Convert daily data to monthly
mapping = {
    'last': ['closing_prices', 'exchange_rates'],
    'mean': ['trading_volume', 'volatility'],
    'max': ['daily_high'],
    'min': ['daily_low']
}

pipeline = (PreprocessPipeline("daily-market-data")
    .resample(frequency='M', mapping=mapping)
    .forward_fill(n=0))  # No gap filling for market data
```

### Cross-Country Panel Analysis
```python
# Standardize panel dataset
pipeline = (PreprocessPipeline("cross-country-macro")
    .drop_static()
    .trim_timeseries(completeness=0.5)
    .backfill(fill='median')
    .knn_iterative_imputer(k=7, normalize_features=True))

# Analyze by development level
pipeline.print_info(['USA', 'GBR', 'DEU'])  # Developed
pipeline.print_info(['BRA', 'CHN', 'IND'])  # Emerging
```
