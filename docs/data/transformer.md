# Data Transformation Documentation

The data transformation utilities in `src/data/transformer.py` provide encoding and sampling methods for machine learning preprocessing. The module includes `DummyEncode` for categorical variable encoding and `DownsampleMajority` for class imbalance correction. These transformers handle unseen categories and maintain consistency between training and test data to prevent data leakage.

## Usage Instructions

```python
from src.data.transformer import DummyEncode, DownsampleMajority

# Dummy encoding for categorical variables
encoder = DummyEncode('Country')
train_encoded = encoder.fit_transform(train_df)
test_encoded = encoder.transform(test_df)

# Downsampling majority class
downsampler = DownsampleMajority(random_state=42)
balanced_df, balanced_labels = downsampler.transform(train_df, train_labels)

# Static method usage
balanced_df, balanced_labels = DownsampleMajority.downsample(
    train_df, train_labels, random_state=42
)
```

## Features

1. **Categorical Encoding**: Robust dummy variable creation with unseen category handling
2. **Class Balancing**: Majority class downsampling for imbalanced datasets
3. **Data Consistency**: Maintains feature structure between train and test sets
4. **Flexible Interface**: Both instance and static method access patterns
5. **Reproducible Sampling**: Random state control for consistent results
6. **Error Prevention**: Validation checks for missing columns and unfitted transformers

## API Reference

### Categorical Encoding

#### `DummyEncode(column_name)`

Encoder for converting categorical variables to dummy variables with proper handling of unseen categories.

**Parameters:**
- `column_name` (str): Name of the categorical column to encode

**Raises:**
- `ValueError`: If specified column not found in DataFrame

**Example:**
```python
# Initialize encoder for Country column
encoder = DummyEncode('Country')

# Fit on training data
train_encoded = encoder.fit_transform(train_df)

# Transform test data (handles unseen countries)
test_encoded = encoder.transform(test_df)
```

**Key Methods:**

##### `fit(df)`

Fit the encoder on training data to learn categories.

**Parameters:**
- `df` (pd.DataFrame): Training DataFrame containing the categorical column

**Returns:**
- `pd.DataFrame`: DataFrame with original columns plus dummy columns

**Raises:**
- `ValueError`: If column not found in DataFrame

**Example:**
```python
import pandas as pd

train_df = pd.DataFrame({
    'Country': ['USA', 'GBR', 'DEU'],
    'GDP': [100, 80, 90]
})

encoder = DummyEncode('Country')
encoded_df = encoder.fit(train_df)
print(encoded_df.columns)
# Output: ['Country', 'GDP', 'Country_DEU', 'Country_GBR', 'Country_USA']
```

##### `transform(df)`

Transform new data using the fitted encoder.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to transform

**Returns:**
- `pd.DataFrame`: DataFrame with original columns plus dummy columns

**Raises:**
- `ValueError`: If encoder not fitted or column not found

**Example:**
```python
# Test data with new and existing countries
test_df = pd.DataFrame({
    'Country': ['USA', 'FRA', 'GBR'],  # FRA is unseen
    'GDP': [102, 85, 82]
})

# Transform using fitted encoder
test_encoded = encoder.transform(test_df)

# FRA gets all dummy columns as 0 (unseen category handling)
print(test_encoded[['Country_DEU', 'Country_GBR', 'Country_USA']].values)
# Output: [[0, 0, 1], [0, 0, 0], [0, 1, 0]]
```

##### `fit_transform(df)`

Fit the encoder and transform data in one step.

**Parameters:**
- `df` (pd.DataFrame): Training DataFrame to fit and transform

**Returns:**
- `pd.DataFrame`: Transformed DataFrame with dummy columns

**Example:**
```python
# Single step fit and transform
encoder = DummyEncode('Country')
encoded_df = encoder.fit_transform(train_df)
```

### Class Balancing

#### `DownsampleMajority(random_state=42)`

Downsampler for balancing datasets by reducing majority class samples.

**Parameters:**
- `random_state` (int, optional): Random seed for reproducible sampling. Defaults to 42

**Example:**
```python
# Initialize downsampler
downsampler = DownsampleMajority(random_state=123)

# Apply to imbalanced data
balanced_df, balanced_labels = downsampler.transform(train_df, train_labels)
```

##### `downsample(df, labels, random_state=42)` [Static Method]

Static method for downsampling majority class to match minority class size.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `labels` (pd.Series or np.array): Binary labels (0 for majority, 1 for minority)
- `random_state` (int, optional): Random seed for reproducible results. Defaults to 42

**Returns:**
- `Tuple[pd.DataFrame, np.array]`: Balanced DataFrame and corresponding labels

**Example:**
```python
import pandas as pd
import numpy as np

# Imbalanced dataset (90% majority class)
df = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000)
})
labels = np.array([0]*900 + [1]*100)  # 900 majority, 100 minority

# Downsample majority class
balanced_df, balanced_labels = DownsampleMajority.downsample(
    df, labels, random_state=42
)

print(f"Original shape: {df.shape}, labels: {np.bincount(labels)}")
print(f"Balanced shape: {balanced_df.shape}, labels: {np.bincount(balanced_labels.astype(int))}")
# Output: Original shape: (1000, 2), labels: [900 100]
#         Balanced shape: (200, 2), labels: [100 100]
```

##### `transform(df, labels)`

Instance method for downsampling using the configured random state.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `labels` (pd.Series or np.array): Binary labels

**Returns:**
- `Tuple[pd.DataFrame, np.array]`: Balanced DataFrame and corresponding labels

**Example:**
```python
# Using instance method
downsampler = DownsampleMajority(random_state=42)
balanced_df, balanced_labels = downsampler.transform(df, labels)
```

## Categorical Encoding Details

### Unseen Category Handling

The `DummyEncode` class handles unseen categories gracefully:

1. **Training Phase**: Records all unique categories from training data
2. **Transform Phase**: Creates dummy columns for all training categories
3. **Unseen Categories**: Assigns zeros to all dummy columns for new categories

**Example:**
```python
# Training data
train_df = pd.DataFrame({'Country': ['USA', 'GBR', 'DEU']})

# Test data with unseen category
test_df = pd.DataFrame({'Country': ['USA', 'FRA', 'JPN']})

encoder = DummyEncode('Country')
encoder.fit(train_df)

# Transform creates columns for training categories only
test_encoded = encoder.transform(test_df)
print(list(test_encoded.columns))
# Output: ['Country', 'Country_DEU', 'Country_GBR', 'Country_USA']

# Unseen categories (FRA, JPN) get all zeros
print(test_encoded.iloc[1:3, -3:].values)  # FRA and JPN rows
# Output: [[0, 0, 0], [0, 0, 0]]
```

### Column Ordering

The encoder maintains consistent column ordering:

- Original columns come first
- Dummy columns are sorted alphabetically by category name
- Order is preserved between fit and transform operations

## Class Balancing Details

### Downsampling Strategy

The `DownsampleMajority` class uses the following approach:

1. **Identify Classes**: Separates majority (0) and minority (1) samples
2. **Random Sampling**: Randomly samples from majority class without replacement
3. **Target Size**: Samples majority class to match minority class size
4. **Recombination**: Concatenates downsampled majority with all minority samples

### Reproducibility

Both static and instance methods support random state control:

```python
# Consistent results with same random state
result1 = DownsampleMajority.downsample(df, labels, random_state=42)
result2 = DownsampleMajority.downsample(df, labels, random_state=42)
# result1 and result2 will be identical

# Instance method uses configured random state
downsampler = DownsampleMajority(random_state=42)
result3 = downsampler.transform(df, labels)
# result3 will match result1 and result2
```

## Error Handling

### Common Error Scenarios

#### Missing Column Error
```python
encoder = DummyEncode('NonExistentColumn')
try:
    encoder.fit(df)  # Raises ValueError
except ValueError as e:
    print(f"Error: {e}")
```

#### Unfitted Transformer Error
```python
encoder = DummyEncode('Country')
try:
    encoder.transform(df)  # Raises ValueError - must fit first
except ValueError as e:
    print(f"Error: {e}")
```

### Data Validation

The transformers include validation checks:
- Column existence verification
- Transformer state validation
- Input data type consistency

## Best Practices

### 1. Categorical Encoding Workflow
```python
# Proper encoding workflow
encoder = DummyEncode('Country')

# Fit only on training data
train_encoded = encoder.fit_transform(train_df)

# Transform test data with same encoder
test_encoded = encoder.transform(test_df)

# Validation data uses same encoder
val_encoded = encoder.transform(validation_df)
```

### 2. Class Balancing Timing
```python
# Apply balancing after encoding but before model training
encoder = DummyEncode('Country')
train_encoded = encoder.fit_transform(train_df)

# Balance the encoded training data
downsampler = DownsampleMajority(random_state=42)
balanced_df, balanced_labels = downsampler.transform(train_encoded, train_labels)

# Don't balance test data - keep original distribution
test_encoded = encoder.transform(test_df)
```

### 3. Reproducible Transformations
```python
# Set random states for reproducibility
RANDOM_STATE = 42

# Use consistent random state across transformations
downsampler = DownsampleMajority(random_state=RANDOM_STATE)
balanced_df, balanced_labels = downsampler.transform(train_df, train_labels)

# Or use static method with explicit random state
balanced_df, balanced_labels = DownsampleMajority.downsample(
    train_df, train_labels, random_state=RANDOM_STATE
)
```

### 4. Feature Engineering Pipeline Integration
```python
# Integrate with preprocessing pipeline
def preprocess_data(train_df, test_df, train_labels, categorical_cols):
    """Complete preprocessing pipeline"""
    
    # 1. Encode categorical variables
    encoders = {}
    for col in categorical_cols:
        encoder = DummyEncode(col)
        train_df = encoder.fit_transform(train_df)
        test_df = encoder.transform(test_df)
        encoders[col] = encoder
    
    # 2. Balance training data only
    downsampler = DownsampleMajority(random_state=42)
    train_balanced, labels_balanced = downsampler.transform(train_df, train_labels)
    
    return train_balanced, test_df, labels_balanced, encoders
```

## Common Use Cases

### Financial Crisis Prediction
```python
# Encode country variable for crisis prediction
crisis_df = pd.DataFrame({
    'Country': ['USA', 'GBR', 'DEU', 'FRA'] * 100,
    'GDP_growth': np.random.randn(400),
    'Crisis': [0, 0, 1, 0] * 100  # Imbalanced: 75% no crisis, 25% crisis
})

# Split data
train_size = int(0.8 * len(crisis_df))
train_df = crisis_df.iloc[:train_size]
test_df = crisis_df.iloc[train_size:]
train_labels = train_df['Crisis'].values
test_labels = test_df['Crisis'].values

# Encode countries
encoder = DummyEncode('Country')
train_encoded = encoder.fit_transform(train_df.drop('Crisis', axis=1))
test_encoded = encoder.transform(test_df.drop('Crisis', axis=1))

# Balance training data (crisis prediction often imbalanced)
downsampler = DownsampleMajority(random_state=42)
train_balanced, labels_balanced = downsampler.transform(train_encoded, train_labels)

print(f"Original training balance: {np.bincount(train_labels)}")
print(f"Balanced training: {np.bincount(labels_balanced.astype(int))}")
```

### Cross-Country Analysis
```python
# Handle multiple categorical variables
country_data = pd.DataFrame({
    'Country': ['USA', 'GBR', 'DEU'] * 50,
    'Region': ['Americas', 'Europe', 'Europe'] * 50,
    'Development': ['Developed', 'Developed', 'Developed'] * 50,
    'GDP': np.random.randn(150)
})

# Encode all categorical variables
categorical_cols = ['Country', 'Region', 'Development']
encoded_df = country_data.copy()

encoders = {}
for col in categorical_cols:
    encoder = DummyEncode(col)
    encoded_df = encoder.fit_transform(encoded_df)
    encoders[col] = encoder

# Remove original categorical columns if needed
for col in categorical_cols:
    encoded_df = encoded_df.drop(col, axis=1)

print(f"Original columns: {list(country_data.columns)}")
print(f"Encoded columns: {list(encoded_df.columns)}")
```

### Time Series Cross-Validation
```python
# Apply same transformations across time-based folds
def create_time_folds(df, n_folds=5):
    """Create time-based cross-validation folds"""
    fold_size = len(df) // n_folds
    folds = []
    
    for i in range(n_folds):
        # Use expanding window for training (all previous data)
        train_end = (i + 1) * fold_size
        val_start = train_end
        val_end = min(train_end + fold_size, len(df))
        
        train_idx = list(range(train_end))
        val_idx = list(range(val_start, val_end))
        
        folds.append((train_idx, val_idx))
    
    return folds

# Apply consistent transformations across folds
folds = create_time_folds(crisis_df)
fold_results = []

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    # Split data for this fold
    fold_train = crisis_df.iloc[train_idx]
    fold_val = crisis_df.iloc[val_idx]
    
    train_labels = fold_train['Crisis'].values
    val_labels = fold_val['Crisis'].values
    
    # Encode categorical variables
    encoder = DummyEncode('Country')
    train_encoded = encoder.fit_transform(fold_train.drop('Crisis', axis=1))
    val_encoded = encoder.transform(fold_val.drop('Crisis', axis=1))
    
    # Balance training data
    if len(np.unique(train_labels)) > 1:  # Only balance if both classes present
        downsampler = DownsampleMajority(random_state=42)
        train_balanced, labels_balanced = downsampler.transform(train_encoded, train_labels)
    else:
        train_balanced, labels_balanced = train_encoded, train_labels
    
    # Store fold results
    fold_results.append({
        'fold': fold_idx,
        'train_shape': train_balanced.shape,
        'val_shape': val_encoded.shape,
        'train_balance': np.bincount(labels_balanced.astype(int)),
        'val_balance': np.bincount(val_labels)
    })

# Analyze fold consistency
for result in fold_results:
    print(f"Fold {result['fold']}: Train {result['train_shape']}, "
          f"Val {result['val_shape']}, Balance {result['train_balance']}")
```
