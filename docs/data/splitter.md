# Data Splitting Documentation

The data splitting utilities in `src/data/splitter.py` provide temporal and geographic data partitioning for financial time series analysis. The `DataSplitter` class handles train-test splitting while maintaining temporal consistency and preventing data leakage. The splitter supports both temporal splitting (chronological order) and geographic splitting (country-based), with algorithms for balanced label distribution and optimal cutoff point selection.

## Usage Instructions

```python
from src.data.splitter import DataSplitter

# Temporal splitting with proportional allocation
splitter = DataSplitter(
    geographic=False,
    train_prop=0.8,
    test_prop=0.2
)

# Geographic splitting by country
splitter = DataSplitter(
    geographic=True,
    train_prop=0.7,
    test_prop=0.3
)

# Custom date range with fixed cutoff
splitter = DataSplitter(
    start_date='2010-01-01',
    end_date='2020-12-31',
    cutoff_date='2018-01-01'
)

# Perform splitting
train_data, y_train = splitter.split(df, 'train')
test_data, y_test = splitter.split(df, 'test')

# Get split information
split_info = splitter.get_split_info(df)
```

## Features

1. **Temporal Consistency**: Maintains chronological order in time series splits
2. **Geographic Partitioning**: Country-based splitting for cross-sectional analysis
3. **Label Balance**: Algorithm for improving label distribution balance across splits
4. **Data Leakage Prevention**: Ensures no future information leaks into training data
5. **Flexible Date Ranges**: Custom start, end, and cutoff date specification
6. **Split Reusability**: Apply same split logic to multiple datasets

## API Reference

### Main Class

#### `DataSplitter(geographic=False, start_date=None, end_date=None, cutoff_date=None, train_prop=0.8, test_prop=0.2)`

Main data splitting class for temporal and geographic partitioning.

**Parameters:**
- `geographic` (bool, optional): Whether to perform geographic splitting. Defaults to False
- `start_date` (Union[str, pd.Timestamp], optional): Start date for data filtering
- `end_date` (Union[str, pd.Timestamp], optional): End date for data filtering  
- `cutoff_date` (Union[str, pd.Timestamp], optional): Fixed cutoff date for temporal splitting
- `train_prop` (float, optional): Training data proportion. Defaults to 0.8
- `test_prop` (float, optional): Test data proportion. Defaults to 0.2

**Raises:**
- `ValueError`: If proportions don't sum to 1.0 or are outside [0.0, 1.0] range
- `TypeError`: If dates cannot be converted to datetime format

**Example:**
```python
# Temporal split with 80-20 proportion
splitter = DataSplitter(train_prop=0.8, test_prop=0.2)

# Geographic split with custom date range
splitter = DataSplitter(
    geographic=True,
    start_date='2015-01-01',
    end_date='2023-12-31',
    train_prop=0.7,
    test_prop=0.3
)
```

### Splitting Methods

#### `split_type()`

Get the type of split being performed.

**Returns:**
- `str`: 'geographic' for country-based splitting, 'temporal' for time-based splitting

**Example:**
```python
splitter = DataSplitter(geographic=True)
print(splitter.split_type())  # Output: 'geographic'
```

#### `perform_split(df, beta=0.8)`

Execute the data splitting operation based on configured parameters.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame with 'Country' and 'Date' columns
- `beta` (float, optional): Weighting factor for label proportion optimization. Defaults to 0.8

**Raises:**
- `TypeError`: If 'Date' column cannot be converted to datetime
- `ValueError`: If fewer than 2 unique labels exist for splitting

**Splitting Logic:**
1. **Fixed Cutoff**: If `cutoff_date` provided, splits at exact date
2. **Geographic**: Groups countries to achieve target proportions with balanced labels
3. **Temporal**: Finds optimal cutoff date balancing temporal and label proportions

**Example:**
```python
splitter = DataSplitter(train_prop=0.8, test_prop=0.2)
splitter.perform_split(df, beta=0.9)  # Higher weight on label balance
```

#### `split(split, df=None, beta=0.8)`

Return training or test split, optionally performing split on new data.

**Parameters:**
- `split` (str): Split to return ('train' or 'test')
- `df` (pd.DataFrame, optional): New DataFrame to split using existing cutoff
- `beta` (float, optional): Label proportion weighting factor. Defaults to 0.8

**Returns:**
- `Tuple[pd.DataFrame, np.array]`: DataFrame and corresponding labels for requested split

**Raises:**
- `ValueError`: If split is not 'train' or 'test', or if no initial data provided

**Example:**
```python
# Initial split
train_data, y_train = splitter.split(df, 'train')
test_data, y_test = splitter.split(df, 'test')

# Apply same split to new data
new_train, new_y_train = splitter.split('train', df=new_df)
```

### Information Methods

#### `get_split_date()`

Get the cutoff date used for temporal splitting.

**Returns:**
- `pd.Timestamp`: Maximum date in training data (temporal split boundary)

**Raises:**
- `ValueError`: If data has not been split yet

**Example:**
```python
splitter.perform_split(df)
cutoff = splitter.get_split_date()
print(f"Split date: {cutoff}")
```

#### `get_split_info(df, beta=0.8)`

Get detailed information about data splitting without performing actual split.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to analyze for splitting
- `beta` (float, optional): Label proportion weighting factor. Defaults to 0.8

**Returns:**
- `dict`: Split information with structure:
  ```python
  {
      'train': {'start': pd.Timestamp, 'end': pd.Timestamp, 'count': int},
      'test': {'start': pd.Timestamp, 'end': pd.Timestamp, 'count': int},
      'total_periods': int,
      'proportions': {'train': float, 'test': float},
      'label_proportions': {'train': float, 'test': float}
  }
  ```

**Example:**
```python
info = splitter.get_split_info(df)
print(f"Training period: {info['train']['start']} to {info['train']['end']}")
print(f"Training samples: {info['train']['count']}")
print(f"Label balance - Train: {info['label_proportions']['train']:.3f}")
```

## Splitting Algorithms

### Temporal Splitting Algorithm

The temporal splitting uses an optimization approach to balance:
1. **Temporal Proportion**: Achieving target train/test proportions
2. **Label Balance**: Maintaining similar label distributions across splits

**Optimization Function:**
```python
score = beta * (label_deviation²) + (1-beta) * (proportion_deviation²)
```

Where:
- `beta`: Weight parameter (0.0-1.0) controlling label vs. proportion importance
- Higher `beta`: Prioritizes label balance over exact proportions
- Lower `beta`: Prioritizes exact proportions over label balance

**Process:**
1. Calculate overall label proportion in dataset
2. Iterate through candidate cutoff dates (middle 50% to 95% of data)
3. For each cutoff, calculate label proportions in train/test splits
4. Select cutoff minimizing combined deviation score

### Geographic Splitting Algorithm

The geographic splitting uses a greedy optimization with local search:

**Phase 1: Greedy Initialization**
1. Randomly shuffle countries (multiple restarts)
2. Add countries sequentially until target sum exceeded
3. Record best initial solution across restarts

**Phase 2: Local Search Improvement**
1. Try single country swaps between train/test sets
2. Accept swaps that reduce distance from target
3. Continue until no improving swaps found

**Early Termination:**
- Stop if within tolerance (default 1% of target)
- Limit improvement iterations to prevent infinite loops

## Split Validation

### Temporal Consistency Checks
- Training data always precedes test data chronologically
- No overlap between training and test periods
- Minimum data requirements (5% of periods for each split)

### Label Distribution Validation
- Both splits contain both classes (0 and 1 labels)
- Warning if extreme label imbalance (>95% single class)
- Balanced allocation of minority class samples

### Data Leakage Prevention
- Future information never used in training splits
- Geographic splits ensure country independence
- Reusable splits maintain consistent cutoff dates

## Configuration Options

### Split Type Selection

#### Temporal Split (Default)
```python
splitter = DataSplitter(geographic=False, train_prop=0.8, test_prop=0.2)
```

**Best For:**
- Time series forecasting
- Crisis prediction models
- Sequential pattern recognition
- When temporal order matters

#### Geographic Split
```python
splitter = DataSplitter(geographic=True, train_prop=0.7, test_prop=0.3)
```

**Best For:**
- Cross-country generalization
- Policy impact analysis
- Country-specific model validation
- When spatial independence required

### Date Range Specification

#### Full Dataset Range
```python
# Uses entire date range available in data
splitter = DataSplitter(train_prop=0.8, test_prop=0.2)
```

#### Custom Date Range
```python
splitter = DataSplitter(
    start_date='2010-01-01',
    end_date='2020-12-31',
    train_prop=0.8,
    test_prop=0.2
)
```

#### Fixed Cutoff Date
```python
splitter = DataSplitter(cutoff_date='2018-01-01')
# Automatically sets geographic=False
```

### Optimization Parameters

#### Label Balance Priority (High β)
```python
# Prioritize balanced labels over exact proportions
splitter.perform_split(df, beta=0.9)
```

#### Proportion Priority (Low β)
```python
# Prioritize exact proportions over label balance
splitter.perform_split(df, beta=0.3)
```

## Error Handling

### Common Error Scenarios

#### Invalid Proportions
```python
# This will raise ValueError
splitter = DataSplitter(train_prop=0.6, test_prop=0.5)  # Sum = 1.1
```

#### Invalid Date Formats
```python
# This will raise TypeError
splitter = DataSplitter(start_date='invalid-date')
```

#### Insufficient Data
```python
# This will raise ValueError if <2 unique labels
df_single_label = df[df['label'] == 0]  # Only one class
splitter.split(df_single_label, 'train')
```

### Graceful Degradation
- Returns empty DataFrames if insufficient data for split
- Warns about extreme label imbalances
- Continues with suboptimal splits if optimization fails

## Best Practices

### 1. Choose Appropriate Split Type
```python
# For time series prediction
temporal_splitter = DataSplitter(geographic=False, train_prop=0.8, test_prop=0.2)

# For cross-country analysis
geographic_splitter = DataSplitter(geographic=True, train_prop=0.7, test_prop=0.3)
```

### 2. Balance Label Distribution
```python
# Check label distribution before splitting
print(f"Overall crisis rate: {(build_labels(df) == 1).mean():.3f}")

# Use higher beta for imbalanced datasets
splitter.perform_split(df, beta=0.8)

# Verify split quality
info = splitter.get_split_info(df)
print(f"Train crisis rate: {info['label_proportions']['train']:.3f}")
print(f"Test crisis rate: {info['label_proportions']['test']:.3f}")
```

### 3. Validate Split Quality
```python
# Get comprehensive split information
info = splitter.get_split_info(df)

# Check temporal coverage
print(f"Training: {info['train']['start']} to {info['train']['end']}")
print(f"Testing: {info['test']['start']} to {info['test']['end']}")

# Verify proportions
print(f"Achieved proportions - Train: {info['proportions']['train']:.3f}")
print(f"Target was: {splitter.train_prop}")
```

### 4. Reuse Split Logic
```python
# Perform initial split
train_data, y_train = splitter.split(df, 'train')
test_data, y_test = splitter.split(df, 'test')

# Apply same split to validation data
val_train, val_y_train = splitter.split('train', df=validation_df)
val_test, val_y_test = splitter.split('test', df=validation_df)
```

## Common Use Cases

### Financial Crisis Prediction
```python
# Temporal split for crisis forecasting
crisis_splitter = DataSplitter(
    geographic=False,
    start_date='1990-01-01',
    end_date='2020-12-31',
    train_prop=0.8,
    test_prop=0.2
)

# Higher weight on label balance due to rare crises
train_data, y_train = crisis_splitter.split(df, 'train', beta=0.9)
test_data, y_test = crisis_splitter.split(df, 'test', beta=0.9)

# Verify crisis distribution
print(f"Training crisis rate: {(y_train == 1).mean():.3f}")
print(f"Test crisis rate: {(y_test == 1).mean():.3f}")
```

### Policy Impact Analysis
```python
# Geographic split for cross-country validation
policy_splitter = DataSplitter(
    geographic=True,
    train_prop=0.7,
    test_prop=0.3
)

# Get split information
info = policy_splitter.get_split_info(df)
print(f"Countries in training: {info['train']['count']} samples")
print(f"Countries in testing: {info['test']['count']} samples")

# Perform split
train_data, y_train = policy_splitter.split(df, 'train')
test_data, y_test = policy_splitter.split(df, 'test')
```

### Backtesting with Walk-Forward
```python
# Multiple temporal splits for backtesting
years = range(2010, 2020)
results = []

for year in years:
    # Create year-specific split
    yearly_splitter = DataSplitter(
        start_date=f'{year-5}-01-01',    # 5 years training
        cutoff_date=f'{year}-01-01',      # Fixed cutoff
        end_date=f'{year+1}-01-01'        # 1 year testing
    )
    
    # Get this year's data
    train_data, y_train = yearly_splitter.split(df, 'train')
    test_data, y_test = yearly_splitter.split(df, 'test')
    
    # Train model and evaluate (pseudo-code)
    # model.fit(train_data, y_train)
    # predictions = model.predict(test_data)
    # results.append(evaluate(predictions, y_test))
```

### Data Quality Assessment
```python
# Analyze split quality across different configurations
configs = [
    {'geographic': False, 'train_prop': 0.8, 'beta': 0.5},
    {'geographic': False, 'train_prop': 0.8, 'beta': 0.9},
    {'geographic': True, 'train_prop': 0.7, 'beta': 0.8}
]

for config in configs:
    splitter = DataSplitter(
        geographic=config['geographic'],
        train_prop=config['train_prop'],
        test_prop=1-config['train_prop']
    )
    
    info = splitter.get_split_info(df, beta=config['beta'])
    
    print(f"\nConfig: {config}")
    print(f"Label balance - Train: {info['label_proportions']['train']:.3f}, "
          f"Test: {info['label_proportions']['test']:.3f}")
    print(f"Achieved proportions - Train: {info['proportions']['train']:.3f}")
```
