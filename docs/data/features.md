# Feature Engineering Documentation

The feature engineering pipeline in `src/data/features.py` provides feature creation methods for financial time series analysis. The `FeaturePipeline` class creates temporal features, statistical indicators, and derived variables while preventing look-ahead bias and data leakage. The pipeline integrates with data splitters to ensure proper train-test separation and supports both geographic and temporal splitting strategies.

## Usage Instructions

```python
from src.data.features import FeaturePipeline
from src.data.splitter import DataSplitter

# Initialize with data splitter
splitter = DataSplitter(df, split_type="temporal", test_size=0.2)
pipeline = FeaturePipeline(df, splitter=splitter)

# Initialize with custom data
pipeline = FeaturePipeline(df)
pipeline.set_data(test_df, split='test')

# Create multiple feature types
slope_features = pipeline.create_slope_features(window=3)
rolling_features = pipeline.create_rolling_features(windows=[6, 12])
lag_features = pipeline.create_lag_features(lags=[1, 3, 6])

# Add features to pipeline
pipeline.add_features(slope_features)
pipeline.add_features(rolling_features)
pipeline.add_features(lag_features)

# Access processed data
X_train = pipeline.X_train
X_test = pipeline.X_test
```

## Features

1. **Temporal Safety**: All features respect chronological order and prevent look-ahead bias
2. **Data Leakage Prevention**: Parameters fitted on training data only
3. **Multiple Feature Types**: Slope, acceleration, rolling stats, lags, momentum, volatility
4. **Split Integration**: Works with geographic and temporal splitting strategies
5. **Temporal Continuity**: Ensures smooth transitions at split boundaries

## API Reference

### Main Class

#### `FeaturePipeline(X, splitter=None)`

Main feature engineering pipeline class for financial time series data.

**Parameters:**
- `X` (pd.DataFrame): Input DataFrame with 'Country' and 'Date' columns
- `splitter` (DataSplitter, optional): DataSplitter instance for train-test separation

**Raises:**
- `ValueError`: If DataFrame lacks required 'Country' or 'Date' columns

**Example:**
```python
# With data splitter
splitter = DataSplitter(df, split_type="temporal", test_size=0.3)
pipeline = FeaturePipeline(df, splitter=splitter)

# Without splitter (uses entire dataset as training)
pipeline = FeaturePipeline(df)
```

### Data Management Methods

#### `set_data(df, split='test')`

Set training or test data for the pipeline.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with 'Country' and 'Date' columns
- `split` (str, optional): Data split to set ('train' or 'test'). Defaults to 'test'

**Raises:**
- `ValueError`: If DataFrame lacks required columns or doesn't match training columns

**Example:**
```python
# Set test data
pipeline.set_data(test_df, split='test')

# Update training data
pipeline.set_data(new_train_df, split='train')
```

#### `combine_temporally()`

Combine training and test data into a single DataFrame for temporal splits.

**Returns:**
- `pd.DataFrame`: Combined DataFrame sorted by Country and Date

**Raises:**
- `ValueError`: If called on non-temporal split types

**Example:**
```python
# For temporal splits only
combined_df = pipeline.combine_temporally()
```

### Derivative Features

#### `create_slope_features(columns=None, window=2, suffix='slope')`

Create slope features measuring rate of change over specified windows.

**Parameters:**
- `columns` (List[str], optional): Columns to compute slopes for. If None, uses all numeric columns
- `window` (int, optional): Window size for slope calculation. Defaults to 2
- `suffix` (str, optional): Suffix for feature names. Defaults to 'slope'

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: Train and test DataFrames with slope features

**Example:**
```python
# Default slope features
train_slope, test_slope = pipeline.create_slope_features()

# Custom slope features
train_slope, test_slope = pipeline.create_slope_features(
    columns=['gdp', 'inflation'], 
    window=4, 
    suffix='trend'
)
```

#### `create_acceleration_features(columns=None, window=2, suffix='acc')`

Create acceleration features measuring second derivatives (slope of slopes).

**Parameters:**
- `columns` (List[str], optional): Columns to compute acceleration for. If None, uses all numeric columns
- `window` (int, optional): Window size for acceleration calculation. Defaults to 2
- `suffix` (str, optional): Suffix for feature names. Defaults to 'acc'

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: Train and test DataFrames with acceleration features

**Example:**
```python
# Create acceleration features
train_acc, test_acc = pipeline.create_acceleration_features(
    columns=['stock_index', 'bond_yields'],
    window=3
)
```

### Rolling Statistics Features

#### `create_rolling_features(columns=None, windows=[3, 6], stats=['mean', 'std', 'min', 'max'], suffix='rolling')`

Generate rolling statistics for specified columns with multiple windows and statistics.

**Parameters:**
- `columns` (List[str], optional): Columns for rolling statistics. If None, uses all numeric columns
- `windows` (Union[int, List[int]], optional): Rolling window sizes. Defaults to [3, 6]
- `stats` (List[str], optional): Statistics to calculate. Defaults to ['mean', 'std', 'min', 'max']
- `suffix` (str, optional): Suffix for feature names. Defaults to 'rolling'

**Available Statistics:** 'mean', 'std', 'min', 'max', 'median', 'skew', 'kurt'

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: Train and test DataFrames with rolling features

**Example:**
```python
# Multiple rolling statistics
train_roll, test_roll = pipeline.create_rolling_features(
    columns=['gdp', 'unemployment'],
    windows=[6, 12, 24],
    stats=['mean', 'std', 'skew']
)

# Simple rolling means
train_roll, test_roll = pipeline.create_rolling_features(
    windows=12,
    stats=['mean']
)
```

### Threshold and Exceedance Features

#### `create_extreme_binary(columns=None, window=24, threshold=0.9, suffix='exceeds')`

Create binary features indicating exceedance of historical thresholds.

**Parameters:**
- `columns` (List[str], optional): Columns for exceedance calculation. If None, uses all numeric columns
- `window` (int, optional): Rolling window for historical reference. Defaults to 24
- `threshold` (float, optional): Threshold percentile for exceedance. Defaults to 0.9
- `suffix` (str, optional): Suffix for feature names. Defaults to 'exceeds'

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: Train and test DataFrames with binary exceedance features

**Example:**
```python
# 90th percentile exceedance over 24 periods
train_binary, test_binary = pipeline.create_extreme_binary(
    columns=['volatility', 'credit_spread'],
    window=24,
    threshold=0.9
)

# 95th percentile exceedance
train_binary, test_binary = pipeline.create_extreme_binary(
    threshold=0.95,
    suffix='extreme'
)
```

### Exponentially Weighted Features

#### `create_exponentially_weighted_averages(columns=None, spans=[3, 6, 12], suffix='ewm')`

Create exponentially weighted moving averages for specified columns.

**Parameters:**
- `columns` (List[str], optional): Columns for EWM calculation. If None, uses all numeric columns
- `spans` (Union[int, List[int]], optional): Span values for EWM. Defaults to [3, 6, 12]
- `suffix` (str, optional): Suffix for feature names. Defaults to 'ewm'

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: Train and test DataFrames with EWM features

**Example:**
```python
# Multiple EWM spans
train_ewm, test_ewm = pipeline.create_exponentially_weighted_averages(
    columns=['prices', 'returns'],
    spans=[6, 12, 24]
)

# Single EWM span
train_ewm, test_ewm = pipeline.create_exponentially_weighted_averages(
    spans=12,
    suffix='smooth'
)
```

### Mean Reversion Features

#### `create_regression_to_mean(columns=None, suffix='dist_mean')`

Create features measuring distance from historical mean for mean reversion analysis.

**Parameters:**
- `columns` (List[str], optional): Columns for mean distance calculation. If None, uses all numeric columns
- `suffix` (str, optional): Suffix for feature names. Defaults to 'dist_mean'

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: Train and test DataFrames with mean reversion features

**Example:**
```python
# Distance from mean features
train_rtm, test_rtm = pipeline.create_regression_to_mean(
    columns=['interest_rates', 'exchange_rates'],
    suffix='mean_dev'
)
```

### Lag Features

#### `create_lag_features(columns=None, lags=[1, 3, 6], suffix='lag')`

Create lag features for specified columns with multiple lag periods.

**Parameters:**
- `columns` (List[str], optional): Columns for lag creation. If None, uses all numeric columns
- `lags` (Union[int, List[int]], optional): Lag periods to create. Defaults to [1, 3, 6]
- `suffix` (str, optional): Suffix for feature names. Defaults to 'lag'

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: Train and test DataFrames with lag features

**Example:**
```python
# Multiple lag periods
train_lags, test_lags = pipeline.create_lag_features(
    columns=['gdp_growth', 'inflation'],
    lags=[1, 2, 4, 8]
)

# Single lag period
train_lags, test_lags = pipeline.create_lag_features(
    lags=3,
    suffix='prev'
)
```

### Momentum Features

#### `create_momentum_features(columns=None, window=3, suffix='momentum')`

Create momentum features measuring directional persistence over windows.

**Parameters:**
- `columns` (List[str], optional): Columns for momentum calculation. If None, uses all numeric columns
- `window` (int, optional): Window size for momentum calculation. Defaults to 3
- `suffix` (str, optional): Suffix for feature names. Defaults to 'momentum'

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: Train and test DataFrames with momentum features

**Example:**
```python
# Create momentum features
train_mom, test_mom = pipeline.create_momentum_features(
    columns=['stock_returns', 'bond_returns'],
    window=6,
    suffix='trend_strength'
)
```

### Volatility Features

#### `create_volatility_features(columns=None, window=3, suffix='volatility')`

Create volatility features measuring variability over specified windows.

**Parameters:**
- `columns` (List[str], optional): Columns for volatility calculation. If None, uses all numeric columns
- `window` (int, optional): Window size for volatility calculation. Defaults to 3
- `suffix` (str, optional): Suffix for feature names. Defaults to 'volatility'

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: Train and test DataFrames with volatility features

**Example:**
```python
# Create volatility features
train_vol, test_vol = pipeline.create_volatility_features(
    columns=['returns', 'prices'],
    window=12,
    suffix='risk'
)
```

### Seasonal Decomposition Features

#### `create_seasonal_decomposition_features(columns=None, model='additive')`

Create seasonal decomposition features separating trend, seasonal, and residual components.

**Parameters:**
- `columns` (List[str], optional): Columns for decomposition. If None, uses all numeric columns
- `model` (str, optional): Decomposition model ('additive' or 'multiplicative'). Defaults to 'additive'

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: Train and test DataFrames with decomposition features

**Raises:**
- `ValueError`: If model is not 'additive' or 'multiplicative'

**Example:**
```python
# Additive seasonal decomposition
train_seasonal, test_seasonal = pipeline.create_seasonal_decomposition_features(
    columns=['gdp', 'employment'],
    model='additive'
)

# Multiplicative decomposition
train_seasonal, test_seasonal = pipeline.create_seasonal_decomposition_features(
    model='multiplicative'
)
```

### Feature Integration

#### `add_features(features)`

Add created features to the existing training and test DataFrames.

**Parameters:**
- `features` (Tuple[pd.DataFrame, pd.DataFrame]): Tuple of train and test feature DataFrames

**Example:**
```python
# Create and add multiple feature types
slope_features = pipeline.create_slope_features()
rolling_features = pipeline.create_rolling_features()
lag_features = pipeline.create_lag_features()

# Add to pipeline
pipeline.add_features(slope_features)
pipeline.add_features(rolling_features)
pipeline.add_features(lag_features)

# Access combined data
X_train_enhanced = pipeline.X_train
X_test_enhanced = pipeline.X_test
```

## Data Leakage Prevention

The pipeline implements several mechanisms to prevent data leakage:

### Parameter Fitting
- All statistics (means, standard deviations, thresholds) are computed on training data only
- Test data transformations use training-derived parameters
- No future information is used in feature creation

### Temporal Continuity
For temporal splits, features are calculated on combined data to ensure smooth transitions:

```python
# Temporal split handling
if self.split_type == "temporal":
    combined_df = self.combine_temporally()
    combined_features = self._calculate_features(combined_df)
    train_features = combined_features[train_indices]
    test_features = combined_features[test_indices]
```

### Look-Ahead Bias Prevention
- Rolling windows only use historical data
- Lag features respect chronological ordering
- All transformations maintain temporal sequence

## Split Type Integration

The pipeline adapts behavior based on data splitting strategy:

### Temporal Splits
- Features calculated on combined data for continuity
- Ensures smooth transitions at temporal boundaries
- Prevents artificial discontinuities in rolling features

### Geographic Splits
- Features calculated separately for train and test
- Maintains independence between country groups
- Prevents cross-country information leakage

## Feature Naming Convention

Features follow a consistent naming pattern:
```
{original_column}_{feature_type}_{parameter}_{statistic}
```

Examples:
- `gdp_slope_3` - GDP slope over 3-period window
- `inflation_rolling_12_mean` - 12-period rolling mean of inflation
- `returns_lag_6` - 6-period lag of returns
- `volatility_ewm_24` - 24-period EWM of volatility

## Best Practices

### 1. Feature Selection Strategy
```python
# Start with basic features
slope_features = pipeline.create_slope_features()
lag_features = pipeline.create_lag_features(lags=[1, 3])

# Add rolling statistics
rolling_features = pipeline.create_rolling_features(
    windows=[6, 12],
    stats=['mean', 'std']
)

# Add specialized features
extreme_features = pipeline.create_extreme_binary(threshold=0.95)
momentum_features = pipeline.create_momentum_features()
```

### 2. Window Size Selection
```python
# Economic data (quarterly) - use longer windows
quarterly_features = pipeline.create_rolling_features(
    windows=[4, 8, 12],  # 1, 2, 3 years
    stats=['mean', 'std']
)

# Financial data (monthly) - use shorter windows
monthly_features = pipeline.create_rolling_features(
    windows=[3, 6, 12],  # 3, 6, 12 months
    stats=['mean', 'std', 'skew']
)
```

### 3. Memory Management
```python
# For large datasets, create features incrementally
features_list = []

# Create features in batches
slope_features = pipeline.create_slope_features()
features_list.append(slope_features)

rolling_features = pipeline.create_rolling_features()
features_list.append(rolling_features)

# Add all features at once
for features in features_list:
    pipeline.add_features(features)
```

### 4. Column-Specific Feature Engineering
```python
# Different features for different variable types
price_columns = ['stock_index', 'bond_prices', 'commodity_prices']
macro_columns = ['gdp', 'inflation', 'unemployment']

# Financial variables - focus on momentum and volatility
price_features = pipeline.create_momentum_features(columns=price_columns)
vol_features = pipeline.create_volatility_features(columns=price_columns)

# Macro variables - focus on trends and cycles
trend_features = pipeline.create_slope_features(columns=macro_columns)
cycle_features = pipeline.create_seasonal_decomposition_features(columns=macro_columns)
```

## Common Use Cases

### Crisis Prediction Feature Set
```python
# Crisis early warning features
pipeline = FeaturePipeline(df, splitter=splitter)

# Trend features
slope_features = pipeline.create_slope_features(window=4)
acc_features = pipeline.create_acceleration_features(window=3)

# Statistical features
rolling_features = pipeline.create_rolling_features(
    windows=[6, 12, 24],
    stats=['mean', 'std', 'skew']
)

# Extreme value features
binary_features = pipeline.create_extreme_binary(
    window=24,
    threshold=0.95
)

# Add all features
for features in [slope_features, acc_features, rolling_features, binary_features]:
    pipeline.add_features(features)
```

### High-Frequency Trading Features
```python
# Short-term trading features
pipeline = FeaturePipeline(daily_data, splitter=splitter)

# Momentum features
momentum_features = pipeline.create_momentum_features(window=5)
volatility_features = pipeline.create_volatility_features(window=10)

# Mean reversion features
rtm_features = pipeline.create_regression_to_mean()

# EWM features for trend following
ewm_features = pipeline.create_exponentially_weighted_averages(spans=[5, 10, 20])
```

### Macro-Economic Analysis
```python
# Long-term economic features
pipeline = FeaturePipeline(quarterly_data, splitter=splitter)

# Trend analysis
slope_features = pipeline.create_slope_features(window=8)  # 2-year trends

# Business cycle features
seasonal_features = pipeline.create_seasonal_decomposition_features()

# Long-term moving averages
rolling_features = pipeline.create_rolling_features(
    windows=[12, 20, 40],  # 3, 5, 10 years
    stats=['mean', 'median']
)

# Historical context
lag_features = pipeline.create_lag_features(lags=[4, 8, 12, 16])  # 1-4 years
```
