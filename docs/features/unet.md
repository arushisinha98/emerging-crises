# U-Net VAE Module

## Overview

The `unet.py` module implements a U-Net architecture combined with Variational Autoencoder (VAE) for time series analysis. This hybrid approach uses skip connections from the U-Net architecture to preserve information flow while maintaining the probabilistic latent space of VAE.

## Purpose

- Combine U-Net skip connections with VAE for better information preservation
- Handle time series data with enhanced reconstruction quality
- Leverage skip connections to maintain fine-grained temporal features
- Provide improved gradient flow through deep recurrent networks
- Enable better reconstruction of sequential patterns

## Main Classes

### RecurrentUNET

PyTorch neural network implementation combining recurrent layers with U-Net skip connections.

#### Key Features
- Multi-layer encoder with skip connections
- U-Net style decoder with symmetric skip connections
- Support for LSTM and GRU recurrent units
- Batch normalization for training stability
- Xavier weight initialization

### TimeSeriesUNET

High-level interface for time series analysis using the RecurrentUNET architecture.

#### Key Features
- Inherits from TimeSeriesVAE for consistent API
- Automatic skip connection management
- Enhanced reconstruction quality through skip connections
- Compatible with existing VAE training pipeline

## Usage Examples

### Basic U-Net VAE Usage

```python
from src.features.unet import TimeSeriesUNET
import pandas as pd
import numpy as np

# Load sequential data
sequential_data = pd.read_csv('time_series_data.csv')
labels = np.array([0, 1, 0, 1, ...])  # Binary labels

# Parameters for U-Net VAE
unet_params = {
    'sequence_length': 30,        # Length of input sequences
    'n_components': 10,           # Latent dimension
    'hidden_size': 64,           # Hidden size for recurrent layers
    'num_layers': 3,             # Number of encoder/decoder layers
    'learning_rate': 1e-3,
    'batch_size': 32,
    'n_epochs': 100,
    'beta': 1.0,
    'block': 'LSTM',             # 'LSTM' or 'GRU'
    'use_layer_norm': True,      # Layer normalization
    'dropout_rate': 0.1          # Dropout for regularization
}

# Initialize U-Net VAE
unet_vae = TimeSeriesUNET(params=unet_params, seed=42)

# Fit to sequential data
unet_vae.fit(sequential_data, labels)

# Transform to latent space
latent_data = unet_vae.transform(sequential_data)
print(f"Latent representation shape: {latent_data.shape}")
```

### Advanced Configuration

```python
# Deep U-Net with multiple skip connections
deep_params = {
    'sequence_length': 50,
    'n_components': 20,
    'hidden_size': 128,          # Larger hidden size
    'num_layers': 5,             # More layers = more skip connections
    'learning_rate': 5e-4,       # Slower learning for stability
    'batch_size': 64,
    'n_epochs': 200,
    'beta': 0.5,                 # Lower beta for better reconstruction
    'block': 'GRU',              # Try GRU instead of LSTM
    'use_layer_norm': True,
    'dropout_rate': 0.2          # More regularization
}

deep_unet = TimeSeriesUNET(params=deep_params, seed=42)
deep_unet.fit(sequential_data, labels)
```

### Reconstruction and Generation

```python
# Reconstruct input sequences
reconstructed = unet_vae.reconstruct(sequential_data)

# Compare reconstruction quality
original_seq = sequential_data.iloc[0]  # First sequence
recon_seq = reconstructed.iloc[0]       # Reconstructed first sequence

# Calculate sequence-wise reconstruction error
import numpy as np
recon_error = np.mean((original_seq - recon_seq) ** 2)
print(f"Sequence reconstruction error: {recon_error}")

# The U-Net architecture should provide better reconstruction
# due to skip connections preserving fine-grained information
```

### Model Architecture Inspection

```python
# The RecurrentUNET model is accessible after fitting
model = unet_vae.model  # Access the underlying PyTorch model

# Inspect the architecture
print("Encoder layers:", len(model.encoder_layers))
print("Skip projections:", len(model.skip_projections_enc))
print("Decoder layers:", len(model.decoder_layers))

# Check parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

### Training Monitoring

```python
# Plot training history
unet_vae.plot_training_history(
    save_path="unet_training.png",
    show=True
)

# Monitor both reconstruction and KL losses
# U-Net should show better reconstruction loss convergence
# due to skip connections helping gradient flow
```

### Comparison with Standard VAE

```python
from src.features.vae import TimeSeriesVAE

# Standard VAE for comparison
standard_params = unet_params.copy()
standard_vae = TimeSeriesVAE(params=standard_params, seed=42)

# Train both models
standard_vae.fit(sequential_data, labels)
unet_vae.fit(sequential_data, labels)

# Compare reconstructions
standard_recon = standard_vae.reconstruct(sequential_data)
unet_recon = unet_vae.reconstruct(sequential_data)

# Calculate reconstruction errors
standard_error = np.mean((sequential_data - standard_recon) ** 2)
unet_error = np.mean((sequential_data - unet_recon) ** 2)

print(f"Standard VAE reconstruction error: {standard_error:.4f}")
print(f"U-Net VAE reconstruction error: {unet_error:.4f}")
print(f"Improvement: {(standard_error - unet_error) / standard_error * 100:.2f}%")
```

## Architecture Details

### Skip Connection Structure

The RecurrentUNET implements a symmetric encoder-decoder with skip connections:

```
Input → Encoder Layer 1 ──┐
           ↓              │
        Encoder Layer 2 ──┼─┐
           ↓              │ │
        Encoder Layer 3 ──┼─┼─┐
           ↓              │ │ │
        Latent Space      │ │ │
           ↓              │ │ │
        Decoder Layer 3 ←─┘ │ │
           ↓                │ │
        Decoder Layer 2 ←───┘ │
           ↓                  │
        Decoder Layer 1 ←─────┘
           ↓
        Output
```

### Key Components

1. **Multi-layer Encoder**: Each layer processes the sequence and stores features
2. **Skip Connections**: Features from encoder layers are passed to corresponding decoder layers
3. **Latent Space**: Standard VAE latent space with mu and logvar
4. **Symmetric Decoder**: Mirrors encoder structure with skip connection inputs
5. **Batch Normalization**: Applied for training stability

## Parameters

### TimeSeriesUNET Constructor
- `params`: Dictionary containing model parameters
- `seed`: Random seed for reproducibility (int, default: 42)  
- `device`: Computing device (str, optional)

### Key Parameters in params Dictionary
- `sequence_length`: Length of input sequences (int)
- `n_components`: Latent space dimensionality (int)
- `hidden_size`: Size of recurrent hidden states (int)
- `num_layers`: Number of encoder/decoder layers (int, affects skip connections)
- `block`: Recurrent unit type ('LSTM' or 'GRU')
- `use_layer_norm`: Whether to use layer normalization (bool, default: True)
- `dropout_rate`: Dropout rate for regularization (float, default: 0.1)
- Standard VAE parameters (learning_rate, batch_size, n_epochs, beta, etc.)

## Methods

### Core Methods
- `fit(data, labels)`: Train the U-Net VAE model
- `transform(data, labels)`: Encode sequences to latent space
- `reconstruct(data)`: Reconstruct sequences through encoder-decoder
- `plot_training_history()`: Visualize training progress

### Inherited Methods
All methods from TimeSeriesVAE are available, providing consistent API.

## Advantages of U-Net Architecture

### Information Preservation
- **Skip Connections**: Direct paths from encoder to decoder preserve fine details
- **Multi-scale Features**: Each layer captures different temporal scales
- **Gradient Flow**: Better gradient propagation through deep networks
- **Reconstruction Quality**: Superior reconstruction of sequential patterns

### Training Benefits
- **Faster Convergence**: Skip connections help training stability
- **Reduced Vanishing Gradients**: Direct connections aid deep network training
- **Better Feature Reuse**: Encoder features directly inform decoder
- **Improved Learning**: More efficient use of network capacity

## When to Use U-Net VAE

### Ideal Use Cases
- **High-quality Reconstruction**: When reconstruction fidelity is critical
- **Long Sequences**: Better handling of long temporal sequences
- **Complex Patterns**: For data with multi-scale temporal features
- **Deep Networks**: When using many encoder/decoder layers

### Comparison with Standard VAE
- **Better Reconstruction**: Skip connections improve reconstruction quality
- **More Parameters**: Larger model size due to skip connections
- **Training Time**: May take longer to train due to complexity
- **Memory Usage**: Higher memory requirements for storing skip connection features

## Dependencies

- torch: PyTorch framework for neural network implementation
- numpy: Numerical computations
- Inherits dependencies from TimeSeriesVAE:
  - scikit-learn: Preprocessing
  - pandas: Data manipulation
  - matplotlib: Visualization

## Best Practices

### Architecture Design
- Start with 3-5 layers for good skip connection benefits
- Balance hidden_size with num_layers for model capacity
- Use layer normalization for training stability
- Apply moderate dropout (0.1-0.2) to prevent overfitting

### Training
- Monitor both reconstruction and KL losses
- Compare with standard VAE baseline
- Use lower beta values if reconstruction is primary goal
- Consider longer training due to increased complexity

### Memory Management
- Be aware of increased memory usage from skip connections
- Reduce batch_size if memory constraints occur
- Consider gradient checkpointing for very deep models

## Typical Workflow

1. **Data Preparation**: Prepare sequential data with appropriate sequence length
2. **Parameter Selection**: Choose architecture depth and skip connection configuration
3. **Baseline Comparison**: Train standard VAE for comparison
4. **U-Net Training**: Train U-Net VAE with monitoring
5. **Quality Assessment**: Compare reconstruction quality between models
6. **Analysis**: Use improved latent representations for downstream tasks
