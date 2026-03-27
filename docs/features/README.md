# Features Module Documentation

This directory contains documentation for all dimensionality reduction and feature extraction modules in the project. Each module provides different approaches to analyzing and transforming high-dimensional data.

## Module Overview

| Module | Purpose | Key Features |
|--------|---------|--------------|
| [base.md](base.md) | Abstract base class for dimensionality reduction | Standardized interface, visualization methods, component selection |
| [pca.md](pca.md) | Principal Component Analysis | Linear dimensionality reduction, explained variance, class-specific models |
| [temporal_pca.md](temporal_pca.md) | Time series PCA analysis | Sliding window PCA, dynamic PCA, temporal pattern detection |
| [tsne.md](tsne.md) | t-SNE visualization | Non-linear embedding, cluster visualization, perplexity analysis |
| [umap.md](umap.md) | UMAP dimensionality reduction | Manifold learning, new data transformation, parameter optimization |
| [vae.md](vae.md) | Variational Autoencoders | Probabilistic latent models, data generation, class-specific VAE |
| [unet.md](unet.md) | U-Net VAE architecture | Skip connections, enhanced reconstruction, time series modeling |
| [utilities.md](utilities.md) | Model management utilities | Model caching, standardized naming, result persistence |

## Quick Start Guide

### 1. Basic Dimensionality Reduction

```python
# PCA for linear dimensionality reduction
from src.features.pca import BasePCA
pca = BasePCA(n_components=2)
pca.fit(data, labels)
transformed = pca.transform(data)

# t-SNE for visualization
from src.features.tsne import BaseTSNE
tsne = BaseTSNE(n_components=2, perplexity=30)
embedding = tsne.fit_transform(data, labels)

# UMAP for flexible dimensionality reduction
from src.features.umap import BaseUMAP
umap_model = BaseUMAP(n_components=2, n_neighbors=15)
umap_model.fit(data, labels)
umap_embedding = umap_model.transform(data)
```

### 2. Time Series Analysis

```python
# Temporal PCA for time series
from src.features.temporal_pca import SlidingWindowPCA
sw_pca = SlidingWindowPCA(name="analysis", window_size=30, step_size=1)
sw_pca.fit(time_series_data)
windows = sw_pca.fit_transform(time_series_data)

# VAE for sequential data
from src.features.vae import TimeSeriesVAE
params = {'sequence_length': 30, 'n_components': 10, 'hidden_size': 64}
ts_vae = TimeSeriesVAE(params=params)
ts_vae.fit(sequential_data, labels)
latent = ts_vae.transform(sequential_data)
```

### 3. Advanced Models

```python
# U-Net VAE for enhanced reconstruction
from src.features.unet import TimeSeriesUNET
unet_params = {'sequence_length': 30, 'n_components': 10, 'num_layers': 3}
unet = TimeSeriesUNET(params=unet_params)
unet.fit(sequential_data, labels)

# Class-specific models for imbalanced data
from src.features.vae import ClassSpecificVAE
cs_vae = ClassSpecificVAE(majority_params=maj_params, minority_params=min_params)
cs_vae.fit(data, labels)
```

### 4. Model Management

```python
# Automatic model caching
from src.features.utilities import load_or_train_model
from src.features.pca import BasePCA

model = load_or_train_model(
    model_class=BasePCA,
    params={'n_components': 10},
    train_data=data,
    train_labels=labels,
    model_dir="artifacts"
)
```

## Module Categories

### Linear Methods
- **PCA**: Principal Component Analysis for variance-based dimensionality reduction
- **Temporal PCA**: Time-aware PCA with sliding windows and dynamic updates

### Non-linear Methods
- **t-SNE**: Neighborhood-based embedding for visualization
- **UMAP**: Manifold learning with global structure preservation
- **VAE**: Probabilistic autoencoders for generative modeling
- **U-Net VAE**: Enhanced VAE with skip connections

### Specialized Methods
- **Class-specific models**: Separate models for different classes
- **Time series models**: Recurrent architectures for sequential data
- **Supervised methods**: Label-aware dimensionality reduction

## Common Workflow Patterns

### 1. Exploratory Analysis
```python
# Start with PCA for baseline
pca = BasePCA(n_components=0.95)  # Keep 95% variance
pca.fit(data, labels)

# Visualize with t-SNE or UMAP
tsne = BaseTSNE(n_components=2)
embedding = tsne.fit_transform(data, labels)
tsne.plot_2D_feature_space(embedding, labels)
```

### 2. Time Series Analysis
```python
# Use sliding window for pattern detection
sw_pca = SlidingWindowPCA(window_size=30, step_size=5)
sw_pca.fit(time_series_data)
sw_pca.plot_evolving_variance()

# Deep learning for complex patterns
ts_vae = TimeSeriesVAE(params={'sequence_length': 30})
ts_vae.fit(sequential_data, labels)
```

### 3. Production Pipeline
```python
# Use utilities for model management
model = load_or_train_model(
    model_class=BaseUMAP,
    params=production_params,
    train_data=data,
    train_labels=labels,
    force_retrain=False  # Use cached if available
)

# Transform new data
new_embedding = model.transform(new_data)
```

## Visualization Features

All modules inherit rich visualization capabilities from the base class:

- **2D/3D scatter plots** with class-based coloring
- **Component analysis** and explained variance plots
- **Parameter optimization** visualizations
- **Training history** monitoring
- **Quality assessment** metrics

## Best Practices

### Model Selection
1. **Start Simple**: Begin with PCA for baseline understanding
2. **Consider Data Type**: Use time series methods for sequential data
3. **Visualization vs Analysis**: Choose t-SNE/UMAP for viz, VAE for analysis
4. **Class Imbalance**: Use class-specific models when needed

### Parameter Tuning
1. **Use Analysis Tools**: Leverage built-in parameter analysis functions
2. **Cross-validate**: Test multiple parameter combinations
3. **Monitor Training**: Watch loss curves and convergence
4. **Quality Metrics**: Use reconstruction error and embedding quality

### Reproducibility
1. **Set Random Seeds**: Use consistent random states
2. **Cache Models**: Leverage utilities for model persistence
3. **Document Parameters**: Save comprehensive metadata
4. **Version Control**: Track parameter changes over time

## Dependencies

Common dependencies across modules:
- **pandas**: Data manipulation and DataFrames
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning utilities and preprocessing
- **matplotlib**: Plotting and visualization
- **torch**: Deep learning framework (for VAE/U-Net models)
- **umap-learn**: UMAP implementation

## Support and Troubleshooting

### Common Issues
- **Memory Errors**: Reduce batch sizes or use PCA preprocessing
- **Convergence Issues**: Adjust learning rates and increase epochs
- **Poor Quality**: Tune hyperparameters using analysis tools
- **Slow Training**: Use GPU acceleration for deep learning models

### Performance Tips
- **Preprocessing**: Standardize data and handle missing values
- **Dimensionality**: Use PCA preprocessing for very high-dimensional data
- **Batch Sizes**: Optimize based on available memory
- **Early Stopping**: Monitor validation loss to prevent overfitting

For detailed usage instructions and examples, refer to the individual module documentation files.
