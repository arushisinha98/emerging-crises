# Neural Network Architectures Module

## Overview

The `architectures.py` module provides deep learning model architectures for financial crisis prediction. It implements feedforward neural networks (FFNN) and LSTM-based classifiers with PyTorch, designed to work with both tabular and sequential data.

## Purpose

- Provide neural network architectures for financial crisis classification
- Support both feedforward and recurrent neural network approaches
- Enable sequence modeling for time series data
- Offer standardized interfaces compatible with sklearn-like workflows
- Include advanced features like attention mechanisms and residual connections

## Main Classes

### FFNNClassifier

A feedforward neural network classifier for tabular data classification.

#### Key Features
- Configurable multi-layer architecture
- Dropout regularization for overfitting prevention
- Automatic data scaling with RobustScaler
- Training history tracking and visualization
- sklearn-compatible interface

### LSTMClassifier

An LSTM-based classifier for sequential data analysis.

#### Key Features
- Multi-layer LSTM architecture
- Optional attention mechanisms
- Residual connections for deep networks
- Sequence-to-sequence modeling capability
- Fine-tuning support for transfer learning

### _LSTMModel (Internal)

Internal PyTorch implementation of the LSTM architecture with advanced features.

#### Key Features
- Multi-layer bidirectional LSTM
- Attention mechanism implementation
- Residual connections between layers
- Dropout and layer normalization
- Proper weight initialization

## Usage Examples

### Basic FFNN Classification

```python
from src.model.architectures import FFNNClassifier
import pandas as pd
import numpy as np

# Load your tabular data
data = pd.read_csv('financial_data.csv')
labels = np.array([0, 1, 0, 1, ...])  # Binary crisis labels

# Define model parameters
params = {
    'hidden_dims': [256, 128, 64, 32],  # Layer sizes
    'dropout_rate': 0.3,                # Dropout probability
    'learning_rate': 1e-3,             # Adam learning rate
    'batch_size': 128,                 # Training batch size
    'n_epochs': 100                    # Training epochs
}

# Initialize and train the model
ffnn = FFNNClassifier(params=params, seed=42)
ffnn.fit(data, labels)

# Make predictions
predictions = ffnn.predict(test_data)
probabilities = ffnn.predict_proba(test_data)

print(f"Predictions shape: {predictions.shape}")
print(f"Probabilities shape: {probabilities.shape}")
```

### LSTM for Sequential Data

```python
from src.model.architectures import LSTMClassifier

# Parameters for LSTM model
lstm_params = {
    'sequence_length': 12,              # Input sequence length
    'lstm_units': 256,                  # LSTM hidden units
    'num_lstm_layers': 3,               # Number of LSTM layers
    'dense_units': [128, 64],           # Dense layer sizes
    'dropout_rate': 0.4,                # Dropout rate
    'use_attention': True,              # Enable attention mechanism
    'use_residual': True,               # Enable residual connections
    'learning_rate': 1e-3,
    'batch_size': 32,
    'n_epochs': 100
}

# Initialize LSTM classifier
lstm = LSTMClassifier(params=lstm_params, seed=42)

# Fit to sequential data (automatically creates sequences)
lstm.fit(sequential_data, labels)

# Predict on new sequential data
lstm_predictions = lstm.predict(test_sequential_data)
lstm_probabilities = lstm.predict_proba(test_sequential_data)
```

### Advanced LSTM Features

```python
# LSTM with prediction indices (for time series validation)
predictions, probabilities, indices = lstm.predict_with_indices(test_data)

# Get aligned labels for evaluation (handles sequence alignment)
aligned_labels = lstm.get_aligned_labels(test_data, true_labels)

# Calculate metrics with proper alignment
from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(aligned_labels, predictions)
f1 = f1_score(aligned_labels, predictions)

print(f"Aligned accuracy: {accuracy:.4f}")
print(f"Aligned F1-score: {f1:.4f}")
```

### Fine-tuning Pre-trained LSTM

```python
# Fine-tune a pre-trained LSTM model
fine_tune_params = {
    'learning_rate': 5e-4,      # Lower learning rate for fine-tuning
    'n_epochs': 50,             # Fewer epochs
    'batch_size': 16            # Smaller batch size
}

# Fine-tune on new data (only trains top 30% of layers)
lstm.fine_tune(
    train_df=new_data,
    y_train=new_labels,
    fine_tune_percent=0.3,      # Only train top 30% of parameters
    fine_tune_params=fine_tune_params
)

# Get fine-tuning history
fine_tune_history = lstm.get_fine_tune_history()
```

### Custom FFNN Architecture

```python
# Deep feedforward network with custom architecture
deep_params = {
    'hidden_dims': [512, 256, 128, 64, 32, 16],  # Deep architecture
    'dropout_rate': 0.5,                          # Higher dropout
    'learning_rate': 5e-4,                        # Lower learning rate
    'batch_size': 64,                            # Medium batch size
    'n_epochs': 200,                             # More epochs
    'criterion': 'focal_loss'                    # Custom loss function
}

deep_ffnn = FFNNClassifier(params=deep_params, seed=42)
deep_ffnn.fit(data, labels)
```

### Model Training with Custom Loss

```python
import torch.nn as nn
from src.model.loss import FocalLoss

# Use focal loss for imbalanced data
params_with_focal = {
    'hidden_dims': [256, 128, 64],
    'dropout_rate': 0.3,
    'learning_rate': 1e-3,
    'batch_size': 128,
    'n_epochs': 100,
    'criterion': FocalLoss(alpha=0.25, gamma=2.0)
}

ffnn_focal = FFNNClassifier(params=params_with_focal, seed=42)
ffnn_focal.fit(data, labels)
```

## Parameters

### FFNNClassifier Parameters
- `hidden_dims`: List of hidden layer sizes (List[int], default: [256, 128, 64, 32])
- `dropout_rate`: Dropout probability (float, default: 0.3)
- `learning_rate`: Learning rate for Adam optimizer (float, default: 1e-3)
- `batch_size`: Training batch size (int, default: 128)
- `n_epochs`: Number of training epochs (int, default: 100)
- `criterion`: Loss function (torch.nn.Module, optional)
- `seed`: Random seed for reproducibility (int, default: 42)
- `device`: Computing device ('cpu', 'cuda', or None for auto)

### LSTMClassifier Parameters
- `sequence_length`: Length of input sequences (int, default: 12)
- `lstm_units`: Number of LSTM hidden units (int, default: 256)
- `num_lstm_layers`: Number of LSTM layers (int, default: 3)
- `dense_units`: Dense layer sizes after LSTM (List[int], default: [128, 64])
- `dropout_rate`: Dropout probability (float, default: 0.4)
- `use_attention`: Enable attention mechanism (bool, default: True)
- `use_residual`: Enable residual connections (bool, default: True)
- `learning_rate`: Learning rate (float, default: 1e-3)
- `batch_size`: Batch size (int, default: 32)
- `n_epochs`: Training epochs (int, default: 100)
- `criterion`: Loss function (torch.nn.Module, default: BCEWithLogitsLoss)

## Methods

### FFNNClassifier Methods
- `fit(train_df, y_train)`: Train the model on tabular data
- `predict(data_df)`: Make binary predictions
- `predict_proba(data_df)`: Get prediction probabilities
- `forward(x)`: Forward pass through the network

### LSTMClassifier Methods
- `fit(train_df, y_train)`: Train the model on sequential data
- `predict(data_df)`: Make predictions on sequences
- `predict_proba(data_df)`: Get prediction probabilities
- `predict_with_indices(data_df)`: Predict with sequence indices
- `get_aligned_labels(data_df, y_true)`: Align labels with predictions
- `fine_tune(train_df, y_train, fine_tune_percent, fine_tune_params)`: Fine-tune model
- `get_fine_tune_history()`: Get fine-tuning training history

## Architecture Details

### FFNN Architecture
```
Input Layer → Hidden Layer 1 → Dropout → Hidden Layer 2 → ... → Output Layer
```

Features:
- ReLU activation functions
- Batch normalization (optional)
- Xavier weight initialization
- Gradient clipping for stability

### LSTM Architecture
```
Input Sequences → Multi-Layer LSTM → Attention → Dense Layers → Output
```

Advanced Features:
- **Bidirectional LSTM**: Processes sequences in both directions
- **Attention Mechanism**: Focuses on important time steps
- **Residual Connections**: Skip connections between LSTM layers
- **Layer Normalization**: Stabilizes training in deep networks

## Key Features

### Data Preprocessing
- **Automatic Scaling**: RobustScaler for handling outliers
- **Sequence Generation**: Automatic sequence creation from DataFrame
- **Missing Data Handling**: Forward fill and zero imputation
- **Date Consistency**: Validates consecutive dates in sequences

### Training Features
- **Validation Split**: Automatic train/validation splitting
- **Early Stopping**: Prevents overfitting (configurable)
- **Gradient Clipping**: Prevents exploding gradients
- **Mixed Precision**: Optional for faster training (GPU)

### Model Management
- **Reproducibility**: Comprehensive seed setting
- **Device Management**: Automatic GPU/CPU selection
- **Model Persistence**: Save and load trained models
- **Training History**: Track metrics and losses

### Advanced Capabilities
- **Transfer Learning**: Fine-tune pre-trained models
- **Attention Visualization**: Understand model focus
- **Feature Extraction**: Extract embeddings from hidden layers
- **Custom Loss Functions**: Support for various loss functions

## Best Practices

### FFNN Guidelines
- Start with moderate layer sizes (64-256 units)
- Use dropout rates between 0.2-0.5
- Monitor validation loss for overfitting
- Scale features appropriately

### LSTM Guidelines
- Use sequence lengths that capture relevant patterns (6-24 months)
- Enable attention for long sequences
- Use residual connections for deep models (>3 layers)
- Lower learning rates for stability

### Training Tips
- Use larger batch sizes for FFNN (64-256)
- Use smaller batch sizes for LSTM (16-64)
- Monitor both training and validation metrics
- Use early stopping to prevent overfitting

## Dependencies

- torch: PyTorch deep learning framework
- scikit-learn: Preprocessing and metrics
- pandas: Data manipulation
- numpy: Numerical computations
- tqdm: Progress bars
- Dataset classes: BasicDataset, SequentialDataset from dataset.py

## Error Handling

The module includes robust error handling for:
- Invalid parameter combinations
- Missing required data columns
- GPU/CPU compatibility issues
- Training convergence problems
- Sequence generation errors

## Typical Workflow

### FFNN Workflow
1. **Data Preparation**: Ensure tabular format with proper columns
2. **Parameter Selection**: Choose architecture and training parameters
3. **Model Training**: Fit model with training data
4. **Evaluation**: Test on held-out data
5. **Prediction**: Use for new data classification

### LSTM Workflow
1. **Sequence Preparation**: Ensure data has Country/Date columns
2. **Sequence Length**: Choose appropriate sequence length
3. **Model Training**: Fit with automatic sequence generation
4. **Alignment**: Use aligned predictions for evaluation
5. **Fine-tuning**: Adapt to new domains if needed
