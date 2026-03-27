# Variational Autoencoder (VAE) Module

## Overview

The `vae.py` module implements Variational Autoencoders for dimensionality reduction and generative modeling. It provides three main implementations: `BaseVAE` for standard tabular data, `TimeSeriesVAE` for sequential data, and `ClassSpecificVAE` for class-aware modeling.

## Purpose

- Learn probabilistic latent representations of data
- Generate new data samples similar to training data
- Perform non-linear dimensionality reduction with reconstruction capabilities
- Handle sequential/time series data with recurrent architectures
- Provide separate models for different classes in imbalanced datasets

## Main Classes

### VAE (Neural Network)

Core PyTorch implementation of the VAE neural network architecture.

#### Key Features
- Encoder-decoder architecture with latent space
- Reparameterization trick for gradient flow
- Configurable hidden layer dimensions
- Proper weight initialization

### BaseVAE

Standard VAE implementation for tabular data dimensionality reduction.

#### Key Features
- Configurable latent dimensions and architecture
- Beta-VAE support for disentangled representations
- Training history tracking and visualization
- Data reconstruction and sample generation capabilities

### TimeSeriesVAE

Recurrent VAE implementation specifically designed for time series data.

#### Key Features
- LSTM/GRU-based encoder-decoder architecture
- Sequence-to-sequence modeling
- KL annealing for stable training
- Sequential data preprocessing

### ClassSpecificVAE

Class-aware VAE that trains separate models for majority and minority classes.

#### Key Features
- Independent VAE models for each class
- Beneficial for imbalanced datasets
- Class-specific latent representations
- Separate model parameter configurations

## Usage Examples

### Basic VAE Usage

```python
from src.features.vae import BaseVAE
import pandas as pd
import numpy as np

# Load your data
data = pd.read_csv('your_data.csv')
labels = np.array([0, 1, 0, 1, ...])  # Binary labels

# Initialize VAE
vae = BaseVAE(
    n_components=10,              # Latent dimension
    hidden_dims=[256, 128, 64],   # Encoder/decoder architecture
    learning_rate=1e-3,
    batch_size=32,
    n_epochs=100,
    beta=1.0,                     # Standard VAE (beta=1)
    seed=42
)

# Fit the model
vae.fit(data, labels)

# Transform data to latent space
latent_data = vae.transform(data)
print(f"Original shape: {data.shape}")
print(f"Latent shape: {latent_data.shape}")
```

### Beta-VAE for Disentangled Representations

```python
# Beta-VAE with higher beta for disentanglement
beta_vae = BaseVAE(
    n_components=20,
    hidden_dims=[512, 256, 128],
    learning_rate=1e-4,
    batch_size=64,
    n_epochs=200,
    beta=5.0,                     # Higher beta for disentanglement
    seed=42
)

beta_vae.fit(data, labels)
latent_repr = beta_vae.transform(data)
```

### Data Reconstruction and Generation

```python
# Reconstruct original data
reconstructed_data = vae.reconstruct(data)

# Calculate reconstruction error
recon_error = np.mean((data - reconstructed_data) ** 2)
print(f"Mean reconstruction error: {recon_error}")

# Generate new samples
new_samples = vae.generate_samples(n_samples=100)
print(f"Generated samples shape: {new_samples.shape}")
```

### Time Series VAE

```python
from src.features.vae import TimeSeriesVAE

# Parameters for time series VAE
ts_params = {
    'sequence_length': 30,        # Length of input sequences
    'n_components': 10,           # Latent dimension
    'hidden_size': 64,           # LSTM hidden size
    'num_layers': 2,             # Number of LSTM layers
    'learning_rate': 1e-3,
    'batch_size': 32,
    'n_epochs': 100,
    'beta': 1.0,
    'block': 'LSTM'              # 'LSTM' or 'GRU'
}

# Initialize time series VAE
ts_vae = TimeSeriesVAE(params=ts_params, seed=42)

# Fit to sequential data (data should include 'label' column)
ts_vae.fit(sequential_data, labels)

# Transform sequences to latent space
latent_sequences = ts_vae.transform(sequential_data)

# Reconstruct sequences
reconstructed_sequences = ts_vae.reconstruct(sequential_data)
```

### Class-Specific VAE

```python
from src.features.vae import ClassSpecificVAE

# Different parameters for majority and minority classes
majority_params = {
    'n_components': 8,
    'hidden_dims': [256, 128],
    'learning_rate': 1e-3,
    'n_epochs': 100,
    'beta': 1.0
}

minority_params = {
    'n_components': 12,           # More capacity for minority class
    'hidden_dims': [512, 256, 128],
    'learning_rate': 5e-4,        # Slower learning
    'n_epochs': 150,              # More epochs
    'beta': 0.5                   # Lower beta for better reconstruction
}

# Initialize class-specific VAE
cs_vae = ClassSpecificVAE(
    majority_params=majority_params,
    minority_params=minority_params,
    seed=42
)

# Fit with labeled data
cs_vae.fit(data, labels)

# Transform (automatically uses appropriate model per class)
latent_data = cs_vae.transform(data, labels)

# Reconstruct using specific model or both
recon_majority = cs_vae.reconstruct(data[labels==0], use_model='majority')
recon_both = cs_vae.reconstruct(data, use_model='both')
```

### Training Monitoring

```python
# Plot training history
vae.plot_training_history(
    save_path="vae_training.png",
    show=True
)

# For time series VAE
ts_vae.plot_training_history(
    save_path="ts_vae_training.png"
)

# For class-specific VAE  
cs_vae.plot_training_history(
    save_path="cs_vae_training.png"
)
```

### Advanced Configuration

```python
# Custom VAE with specific architecture
custom_vae = BaseVAE(
    n_components=15,
    hidden_dims=[1024, 512, 256, 128],  # Deeper network
    learning_rate=5e-4,
    batch_size=128,                     # Larger batches
    n_epochs=300,                       # More training
    beta=2.0,                          # Moderate disentanglement
    seed=42
)

# Advanced time series VAE with KL annealing
advanced_params = {
    'sequence_length': 50,
    'n_components': 20,
    'hidden_size': 128,
    'num_layers': 3,
    'learning_rate': 1e-3,
    'batch_size': 64,
    'n_epochs': 200,
    'beta': 1.0,
    'block': 'LSTM',
    'kl_anneal_epochs': 50,            # KL annealing period
    'dropout_rate': 0.2                # Regularization
}

advanced_ts_vae = TimeSeriesVAE(params=advanced_params)
```

## Parameters

### BaseVAE Constructor
- `n_components`: Latent space dimensionality (int)
- `hidden_dims`: List of hidden layer sizes (List[int])
- `learning_rate`: Optimization learning rate (float, default: 1e-3)
- `batch_size`: Training batch size (int, default: 32)
- `n_epochs`: Number of training epochs (int, default: 100)
- `beta`: Weight for KL divergence loss (float, default: 1.0)
- `seed`: Random seed for reproducibility (int, default: 42)
- `device`: Computing device ('cpu', 'cuda', or None for auto)

### TimeSeriesVAE Parameters (via params dict)
- `sequence_length`: Length of input sequences (int)
- `n_components`: Latent dimension (int)
- `hidden_size`: LSTM/GRU hidden size (int)
- `num_layers`: Number of recurrent layers (int, default: 1)
- `block`: Recurrent unit type ('LSTM' or 'GRU')
- Additional parameters from BaseVAE

### ClassSpecificVAE Constructor
- `majority_params`: Parameters for majority class model (dict)
- `minority_params`: Parameters for minority class model (dict)
- `seed`: Random seed (int, default: 42)
- `device`: Computing device (str, optional)

## Methods

### BaseVAE Methods
- `fit(data, labels)`: Train the VAE model
- `transform(data, labels)`: Encode data to latent space
- `reconstruct(data)`: Reconstruct data through encoder-decoder
- `generate_samples(n_samples)`: Generate new data samples
- `plot_training_history()`: Visualize training progress
- `get_feature_names()`: Get input feature names
- `get_component_names()`: Get latent component names

### TimeSeriesVAE Methods
- `fit(data, labels)`: Train on sequential data
- `transform(data, labels)`: Transform sequences to latent space  
- `reconstruct(data)`: Reconstruct sequences
- `plot_training_history()`: Plot training curves
- Model automatically handles sequential data preprocessing

### ClassSpecificVAE Methods
- `fit(data, labels)`: Train separate models for each class
- `transform(data, labels)`: Transform using class-appropriate model
- `reconstruct(data, use_model)`: Reconstruct using specific or both models
- `get_model_parameters()`: Get parameters for both models
- `plot_training_history()`: Plot training for both models

## Key Features

### Probabilistic Modeling
- **Latent Variable Model**: Learn probabilistic latent representations
- **Reparameterization Trick**: Enable gradient-based optimization
- **Generative Capabilities**: Sample new data from learned distribution
- **Uncertainty Quantification**: Probabilistic latent space

### Training Stability
- **Beta-VAE**: Control reconstruction vs. regularization trade-off
- **KL Annealing**: Gradual introduction of KL regularization
- **Proper Initialization**: Xavier/Kaiming weight initialization
- **Training Monitoring**: Loss curves and convergence tracking

### Architecture Flexibility
- **Configurable Depth**: Adjustable number of hidden layers
- **Recurrent Support**: LSTM/GRU for sequential data
- **Class-Specific Models**: Separate architectures per class
- **Batch Normalization**: Optional normalization layers

### Data Handling
- **Automatic Preprocessing**: Built-in standardization and scaling
- **Sequential Support**: Handle time series and sequence data
- **Missing Data**: Robust handling of incomplete data
- **Class Imbalance**: Specialized models for imbalanced datasets

## Dependencies

- torch: PyTorch deep learning framework
- scikit-learn: Preprocessing utilities
- pandas: Data manipulation
- numpy: Numerical computations
- matplotlib: Visualization
- Base module: Inherits from DimensionalityReduction abstract class

## Best Practices

### Architecture Design
- Start with moderate latent dimensions (5-20)
- Use deeper networks for complex data
- Adjust beta parameter based on reconstruction vs. disentanglement needs
- Consider class-specific models for imbalanced data

### Training
- Monitor both reconstruction and KL loss
- Use KL annealing for stable training
- Adjust learning rate based on convergence
- Save model checkpoints for long training runs

### Evaluation
- Check reconstruction quality on test data
- Visualize latent space structure
- Generate samples to assess model quality
- Compare different beta values for optimal balance

## Typical Workflow

1. **Data Preparation**: Load and preprocess your dataset
2. **Parameter Selection**: Choose architecture and training parameters
3. **Model Training**: Fit VAE model with monitoring
4. **Quality Assessment**: Evaluate reconstruction and generation quality
5. **Latent Analysis**: Explore learned latent representations
6. **Application**: Use for dimensionality reduction, generation, or analysis
