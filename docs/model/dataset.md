# Dataset Classes for PyTorch Models

## Overview

The `dataset.py` module provides PyTorch Dataset classes for different data types used in financial crisis prediction. It includes datasets for basic tabular data and sophisticated sequential data handling with automatic sequence generation and validation.

## Purpose

- Provide PyTorch-compatible dataset classes for various data formats
- Handle automatic sequence generation from time series data
- Ensure temporal consistency and data integrity
- Support proper scaling and preprocessing pipelines
- Enable efficient batch loading for neural network training

## Main Classes

### BasicDataset

Simple dataset class for tabular data with features and binary targets.

#### Key Features
- Basic tensor conversion for numerical data
- Compatible with standard PyTorch DataLoader
- Memory efficient for large datasets
- Simple interface for feedforward networks

### SequentialDataset

Advanced dataset class for time series data with automatic sequence generation.

#### Key Features
- Automatic sequence creation from multi-index DataFrame
- Temporal consistency validation
- Country-specific sequence processing
- Robust data preprocessing and scaling
- Index tracking for evaluation alignment

## Usage Examples

### Basic Tabular Dataset

```python
from src.model.dataset import BasicDataset
from torch.utils.data import DataLoader
import numpy as np
import torch

# Prepare tabular data
X_data = np.random.randn(1000, 50)  # 1000 samples, 50 features
y_data = np.random.randint(0, 2, 1000)  # Binary labels

# Create dataset
dataset = BasicDataset(data=X_data, target=y_data)

# Basic dataset information
print(f"Dataset size: {len(dataset)}")
print(f"Input dimension: {dataset.get_input_dim()}")

# Create data loader
data_loader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=2
)

# Iterate through batches
for batch_idx, (features, targets) in enumerate(data_loader):
    print(f"Batch {batch_idx}: Features {features.shape}, Targets {targets.shape}")
    if batch_idx == 2:  # Show first 3 batches
        break
```

### Sequential Dataset from DataFrame

```python
from src.model.dataset import SequentialDataset
import pandas as pd
import numpy as np

# Load financial time series data with Country and Date columns
data = pd.read_csv('financial_time_series.csv')
labels = np.array([0, 1, 0, 1, ...])  # Crisis labels

# Create sequential dataset
seq_dataset = SequentialDataset(
    train_df=data,
    y_train=labels,
    sequence_length=12,        # 12-month sequences
    fit_scaler=True           # Fit new scaler
)

print(f"Number of sequences: {len(seq_dataset)}")
print(f"Number of features per timestep: {seq_dataset.n_features}")
print(f"Feature columns: {seq_dataset.feature_cols}")

# Get a sample sequence
sequence, label = seq_dataset[0]
print(f"Sequence shape: {sequence.shape}")  # (sequence_length, n_features)
print(f"Label: {label}")
```

### Advanced Sequential Dataset Usage

```python
# Advanced configuration with existing scaler
from sklearn.preprocessing import RobustScaler

# Pre-fitted scaler for consistent preprocessing
existing_scaler = RobustScaler()
existing_scaler.fit(training_features)

# Create dataset with existing scaler (for test/validation data)
test_dataset = SequentialDataset(
    train_df=test_data,
    y_train=test_labels,
    sequence_length=12,
    scaler=existing_scaler,    # Use existing scaler
    fit_scaler=False          # Don't refit scaler
)

# Access dataset properties
countries = test_dataset.get_countries()
valid_indices = test_dataset.get_valid_indices()

print(f"Countries in dataset: {np.unique(countries)}")
print(f"Valid indices shape: {valid_indices.shape}")
```

### Creating DataLoaders for Training

```python
from torch.utils.data import DataLoader, random_split

# Split sequential dataset for training/validation
train_size = int(0.8 * len(seq_dataset))
val_size = len(seq_dataset) - train_size

train_subset, val_subset = random_split(
    seq_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Create data loaders
train_loader = DataLoader(
    train_subset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    drop_last=False
)

val_loader = DataLoader(
    val_subset,
    batch_size=64,           # Larger batch for validation
    shuffle=False,           # No shuffling for validation
    num_workers=4
)

# Training loop example
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (sequences, targets) in enumerate(train_loader):
        sequences = sequences.to(device)  # Shape: (batch_size, seq_len, features)
        targets = targets.to(device)      # Shape: (batch_size,)
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Country-Based Splitting

```python
# Split by countries for geographical cross-validation
countries = seq_dataset.get_countries()
unique_countries = np.unique(countries)

# Split countries into train/test
train_countries = unique_countries[:int(0.8 * len(unique_countries))]
test_countries = unique_countries[int(0.8 * len(unique_countries)):]

# Create country-based masks
train_mask = np.isin(countries, train_countries)
test_mask = np.isin(countries, test_countries)

# Get indices
train_indices = np.where(train_mask)[0]
test_indices = np.where(test_mask)[0]

# Create subset datasets
from torch.utils.data import Subset

train_geo_dataset = Subset(seq_dataset, train_indices)
test_geo_dataset = Subset(seq_dataset, test_indices)

print(f"Train countries: {train_countries}")
print(f"Test countries: {test_countries}")
print(f"Train sequences: {len(train_geo_dataset)}")
print(f"Test sequences: {len(test_geo_dataset)}")
```

### Data Preprocessing Pipeline

```python
# Complete preprocessing pipeline
def create_sequential_datasets(train_df, test_df, train_labels, test_labels, 
                              sequence_length=12):
    """Create properly scaled train/test sequential datasets"""
    
    # Create training dataset (fits scaler)
    train_dataset = SequentialDataset(
        train_df=train_df,
        y_train=train_labels,
        sequence_length=sequence_length,
        fit_scaler=True  # Fit scaler on training data
    )
    
    # Create test dataset (uses training scaler)
    test_dataset = SequentialDataset(
        train_df=test_df,
        y_train=test_labels,
        sequence_length=sequence_length,
        scaler=train_dataset.scaler,  # Use same scaler
        fit_scaler=False             # Don't refit
    )
    
    return train_dataset, test_dataset

# Usage
train_seq_dataset, test_seq_dataset = create_sequential_datasets(
    train_df, test_df, train_labels, test_labels
)
```

## Parameters

### BasicDataset Constructor
- `data`: Input features as numpy array or tensor-convertible format
- `target`: Target labels as numpy array or tensor-convertible format

### SequentialDataset Constructor
- `train_df`: DataFrame with Country, Date, and feature columns
- `y_train`: Array of labels corresponding to DataFrame rows
- `sequence_length`: Number of timesteps in each sequence (int, default: 12)
- `scaler`: Pre-fitted scaler object (optional, default: None creates RobustScaler)
- `fit_scaler`: Whether to fit the scaler on this data (bool, default: True)

## Methods

### BasicDataset Methods
- `__len__()`: Return number of samples
- `__getitem__(idx)`: Get sample at index (returns features, target tuple)
- `get_input_dim()`: Return input feature dimension

### SequentialDataset Methods
- `__len__()`: Return number of sequences generated
- `__getitem__(idx)`: Get sequence at index (returns sequence tensor, label tensor)
- `get_countries()`: Return array of country labels for each sequence
- `get_valid_indices()`: Return original DataFrame indices for each sequence

## Key Features

### Automatic Sequence Generation
The SequentialDataset automatically creates sequences from time series data:

1. **Country Grouping**: Processes each country separately
2. **Date Sorting**: Ensures chronological order within countries
3. **Consecutive Validation**: Checks for temporal consistency
4. **Sequence Creation**: Generates overlapping sequences of specified length
5. **Label Assignment**: Associates each sequence with the label of its final timestep

### Data Preprocessing
- **Missing Value Handling**: Forward fill followed by zero imputation
- **Scaling**: RobustScaler for handling outliers in financial data
- **Infinite Value Handling**: Replaces inf/-inf with NaN before forward filling
- **Feature Selection**: Automatically excludes non-numeric and metadata columns

### Temporal Consistency Validation
The `_are_dates_consecutive()` method ensures sequences contain consecutive time periods:

- **Daily Data**: Tolerance of 1 day
- **Monthly Data**: Tolerance of 3 days (28-31 day months)
- **Quarterly Data**: Tolerance of 3 days (89-92 day quarters)
- **Yearly Data**: Tolerance of 1 day (365-366 day years)
- **Other Frequencies**: Adaptive tolerance based on detected frequency

### Index Tracking
- **Valid Indices**: Track original DataFrame indices for each sequence
- **Country Labels**: Maintain country information for geographical analysis
- **Evaluation Alignment**: Enable proper alignment of predictions with ground truth

## Advanced Features

### Memory Efficiency
- **Lazy Loading**: Sequences generated on-demand during iteration
- **Efficient Storage**: Uses numpy arrays for internal storage
- **Batch Processing**: Optimized for PyTorch DataLoader integration

### Robustness
- **Error Handling**: Graceful handling of missing or invalid data
- **Data Validation**: Comprehensive checks for data integrity
- **Flexible Input**: Accepts various DataFrame formats and structures

### Integration
- **PyTorch Compatibility**: Full integration with PyTorch ecosystem
- **Sklearn Integration**: Uses sklearn scalers and preprocessors
- **Custom Workflows**: Easy integration with custom training pipelines

## Best Practices

### Sequence Length Selection
- **Short Sequences (6-12 months)**: For capturing immediate crisis indicators
- **Medium Sequences (12-24 months)**: For balanced local and global patterns
- **Long Sequences (24+ months)**: For long-term economic cycle analysis

### Scaling Strategy
- Always fit scaler on training data only
- Use same scaler for validation and test data
- RobustScaler recommended for financial data with outliers
- Consider feature-specific scaling for different variable types

### Data Splitting
- **Temporal Split**: Respect chronological order in train/test split
- **Geographical Split**: Use country-based splitting for robustness
- **Stratified Split**: Maintain crisis/non-crisis proportions

### Memory Management
- Use appropriate batch sizes based on available memory
- Consider sequence length impact on memory usage
- Use num_workers for parallel data loading
- Monitor memory usage during training

## Dependencies

- torch: PyTorch tensors and dataset interface
- pandas: DataFrame operations and time series handling
- numpy: Numerical computations and array operations
- scikit-learn: RobustScaler and preprocessing utilities

## Error Handling

The module handles various error conditions:
- Missing Country or Date columns
- Non-consecutive date sequences
- Invalid sequence lengths
- Missing or infinite values
- Empty datasets or invalid indices

## Performance Considerations

### Optimization Tips
- **Batch Size**: Larger batches for better GPU utilization
- **Num Workers**: Use multiple workers for data loading
- **Pin Memory**: Enable for GPU training
- **Persistent Workers**: Reduce worker startup overhead

### Memory Usage
- Sequence length × number of features × batch size determines memory usage
- Consider gradient accumulation for large effective batch sizes
- Monitor GPU memory during training
- Use mixed precision training if supported

## Integration Examples

### With Custom Training Loop
```python
# Custom training with sequential data
for epoch in range(num_epochs):
    for batch_idx, (sequences, targets) in enumerate(train_loader):
        # sequences: (batch_size, sequence_length, n_features)
        # targets: (batch_size,)
        
        # Your model training code here
        pass
```

### With Lightning Framework
```python
import pytorch_lightning as pl

class FinancialCrisisModule(pl.LightningModule):
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
    
    def training_step(self, batch, batch_idx):
        sequences, targets = batch
        # Your training logic here
        return loss
```
