# Feature Utilities Module

## Overview

The `utilities.py` module provides utility functions for managing trained models, including saving, loading, and caching functionality. It streamlines the workflow of training and reusing dimensionality reduction models across different experiments.

## Purpose

- Provide model persistence and caching functionality
- Generate standardized model names based on parameters
- Implement smart model loading with fallback to training
- Save and load model metadata and training results
- Streamline experiment workflows and reproducibility

## Key Functions

### Model Management Functions

- `model_exists()`: Check if a trained model already exists
- `load_or_train_model()`: Load existing model or train new one
- `save_trained_model()`: Save model with metadata
- `load_trained_model()`: Load model with optional metadata
- `generate_model_name()`: Create standardized model names

### Result Management Functions

- `save_model_results()`: Save training results and transformed data
- `load_model_results()`: Load saved model results
- `get_artifacts_path()`: Get absolute path to artifacts directory

## Usage Examples

### Basic Model Caching

```python
from src.features.utilities import load_or_train_model
from src.features.pca import BasePCA
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('training_data.csv')
labels = np.array([0, 1, 0, 1, ...])

# Define model parameters
params = {
    'n_components': 10,
    'random_state': 42
}

# Load existing model or train new one
pca_model = load_or_train_model(
    model_class=BasePCA,
    params=params,
    train_data=data,
    train_labels=labels,
    model_dir="artifacts",      # Save/load directory
    force_retrain=False         # Set True to force retraining
)

# Model is ready to use (either loaded or freshly trained)
transformed_data = pca_model.transform(data)
```

### Advanced Model Management

```python
from src.features.utilities import (
    model_exists, generate_model_name, 
    save_trained_model, load_trained_model
)

# Generate standardized model name
model_name = generate_model_name(
    model_type="BaseVAE",
    params={
        'n_components': 10,
        'hidden_dims': [256, 128, 64],
        'learning_rate': 1e-3,
        'batch_size': 32,
        'n_epochs': 100,
        'beta': 1.0
    },
    timestamp=False,        # Don't include timestamp
    hash_params=True        # Include parameter hash for uniqueness
)
print(f"Generated name: {model_name}")
# Output: BaseVAE_comp10_hid256-128-64_lr1e-3_bs32_ep100_beta1.0_hash12345

# Check if model exists
if model_exists(model_name, "artifacts"):
    print("Model exists, loading...")
    model, metadata = load_trained_model(
        model_path=f"artifacts/{model_name}.pkl",
        load_metadata=True
    )
else:
    print("Model doesn't exist, training...")
    # Train your model here
    # model = train_my_model(...)
    
    # Save with metadata
    save_trained_model(
        model=model,
        model_name=model_name,
        save_dir="artifacts",
        metadata={
            'training_params': params,
            'dataset_info': 'financial_crisis_data',
            'performance_metrics': {'accuracy': 0.95}
        }
    )
```

### Working with Different Model Types

```python
from src.features.vae import TimeSeriesVAE
from src.features.umap import BaseUMAP

# VAE model caching
vae_params = {
    'sequence_length': 30,
    'n_components': 10,
    'hidden_size': 64,
    'num_layers': 2,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'n_epochs': 100,
    'beta': 1.0,
    'block': 'LSTM'
}

vae_model = load_or_train_model(
    model_class=TimeSeriesVAE,
    params=vae_params,
    train_data=sequential_data,
    train_labels=labels,
    model_dir="artifacts/vae_models"
)

# UMAP model caching
umap_params = {
    'n_components': 2,
    'n_neighbors': 15,
    'min_dist': 0.1,
    'metric': 'euclidean'
}

umap_model = load_or_train_model(
    model_class=BaseUMAP,
    params=umap_params,
    train_data=data,
    train_labels=labels,
    model_dir="artifacts/umap_models"
)
```

### Saving and Loading Results

```python
from src.features.utilities import save_model_results, load_model_results

# Save comprehensive model results
result_paths = save_model_results(
    model=trained_model,
    train_data=train_df,
    test_data=test_df,
    train_labels=train_labels,
    test_labels=test_labels,
    results_dir="artifacts/results"
)

print("Saved results to:")
for key, path in result_paths.items():
    print(f"  {key}: {path}")

# Load saved results
results = load_model_results(
    results_dir="artifacts/results",
    model_name=model_name,
    timestamp="2024-03-15_14-30-25"  # Optional specific timestamp
)

train_transformed = results['train_transformed']
test_transformed = results['test_transformed']
training_history = results['training_history']
```

### Batch Model Operations

```python
# Train multiple models with different parameters
param_grid = [
    {'n_components': 5, 'beta': 0.1},
    {'n_components': 10, 'beta': 0.5},
    {'n_components': 20, 'beta': 1.0},
    {'n_components': 30, 'beta': 5.0}
]

trained_models = {}
for i, params in enumerate(param_grid):
    # Add common parameters
    full_params = {
        **params,
        'learning_rate': 1e-3,
        'batch_size': 32,
        'n_epochs': 100
    }
    
    model = load_or_train_model(
        model_class=BaseVAE,
        params=full_params,
        train_data=data,
        train_labels=labels,
        model_dir="artifacts/grid_search"
    )
    
    model_name = generate_model_name("BaseVAE", full_params)
    trained_models[model_name] = model

print(f"Trained {len(trained_models)} models")
```

### Working with Artifacts Directory

```python
from src.features.utilities import get_artifacts_path
import os

# Get artifacts directory path
artifacts_dir = get_artifacts_path()
print(f"Artifacts directory: {artifacts_dir}")

# List all saved models
model_files = [f for f in os.listdir(artifacts_dir) if f.endswith('.pkl')]
print(f"Available models: {len(model_files)}")
for model_file in model_files[:5]:  # Show first 5
    print(f"  {model_file}")

# Create subdirectories for organization
vae_dir = artifacts_dir / "vae_models"
pca_dir = artifacts_dir / "pca_models"
vae_dir.mkdir(exist_ok=True)
pca_dir.mkdir(exist_ok=True)
```

## Function Parameters

### load_or_train_model()
- `model_class`: Class of the model to train
- `params`: Dictionary of model parameters
- `train_data`: Training data DataFrame
- `train_labels`: Training labels array
- `model_dir`: Directory to save/load models (default: "artifacts")
- `force_retrain`: Force retraining even if model exists (default: False)

### generate_model_name()
- `model_type`: Type of model (e.g., 'VAE', 'PCA', 'UMAP')
- `params`: Dictionary of model parameters
- `timestamp`: Include timestamp in name (default: False)
- `hash_params`: Include parameter hash for uniqueness (default: False)

### save_trained_model()
- `model`: Trained model object
- `model_name`: Name for the saved model
- `save_dir`: Directory to save the model (default: "artifacts")
- `metadata`: Additional metadata dictionary (optional)
- `save_state_dict_only`: Save only PyTorch state dict (default: False)

### load_trained_model()
- `model_path`: Path to the saved model file
- `load_metadata`: Whether to return metadata (default: True)
- `device`: Device for PyTorch models (optional)

## Model Name Generation

The `generate_model_name()` function creates standardized names based on model parameters:

### Standard Format
```
ModelType_param1_param2_param3...
```

### Example Mappings
- `n_components=10` → `comp10`
- `hidden_size=64` → `hid64`  
- `learning_rate=1e-3` → `lr1e-3`
- `batch_size=32` → `bs32`
- `n_epochs=100` → `ep100`
- `beta=1.0` → `beta1.0`
- `num_layers=3` → `lay3`

### Example Output
```python
params = {
    'n_components': 10,
    'hidden_size': 64,
    'num_layers': 3,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'n_epochs': 100,
    'beta': 1.0
}

name = generate_model_name("TimeSeriesVAE", params)
# Result: TimeSeriesVAE_comp10_hid64_lay3_lr1e-3_bs32_ep100_beta1.0
```

## Features

### Automatic Caching
- **Smart Loading**: Automatically loads existing models or trains new ones
- **Parameter Matching**: Uses parameter-based naming for cache hits
- **Fallback Training**: Seamlessly trains if model doesn't exist
- **Force Retrain**: Option to retrain even if cached model exists

### Metadata Management
- **Rich Metadata**: Save training parameters, dataset info, performance metrics
- **Automatic Metadata**: Model type, timestamp, component names automatically added
- **Flexible Storage**: Support for arbitrary metadata dictionaries
- **Easy Retrieval**: Load models with or without metadata

### File Organization
- **Standardized Naming**: Consistent, readable model names
- **Directory Structure**: Organized storage in artifacts directory
- **Subdirectory Support**: Custom subdirectories for different model types
- **Cross-platform Paths**: Proper path handling across operating systems

## Best Practices

### Model Naming
- Use descriptive model types (e.g., "BetaVAE" instead of just "VAE")
- Include key distinguishing parameters in names
- Use timestamps for experimental runs
- Use parameter hashes for exact reproducibility

### Metadata
- Always include training parameters in metadata
- Add dataset information for traceability
- Include performance metrics for model selection
- Store preprocessing information (scalers, etc.)

### Directory Organization
```
artifacts/
├── pca_models/
├── vae_models/
├── umap_models/
├── results/
│   ├── train_data/
│   ├── test_data/
│   └── training_history/
└── experiments/
    ├── experiment_1/
    └── experiment_2/
```

## Dependencies

- pathlib: Path handling utilities
- pickle: Model serialization
- datetime: Timestamp generation
- hashlib: Parameter hashing
- pandas: Data manipulation
- numpy: Array operations
- Project logging utilities

## Error Handling

The module includes robust error handling for:
- Missing model files
- Corrupt model files  
- Invalid parameter combinations
- Directory creation failures
- Serialization errors

All functions provide informative error messages and logging for debugging.
