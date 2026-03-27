# UMAP (Uniform Manifold Approximation and Projection) Module

## Overview

The `umap.py` module implements UMAP for dimensionality reduction and visualization. UMAP excels at preserving both local and global data structure, making it superior to t-SNE for many applications while also supporting transformation of new data points.

## Purpose

- Perform non-linear dimensionality reduction with superior structure preservation
- Enable both visualization and general-purpose dimensionality reduction
- Transform new data points using a fitted model (unlike t-SNE)
- Provide faster computation than t-SNE for large datasets
- Support supervised and unsupervised dimensionality reduction

## Main Classes

### BaseUMAP

UMAP implementation for dimensionality reduction and visualization with optional PCA preprocessing.

#### Key Features
- Configurable embedding dimensions (any positive integer)
- Adjustable neighborhood and distance parameters
- Optional PCA preprocessing for noise reduction
- Support for supervised learning with labels
- Built-in parameter analysis tools
- Ability to transform new data points

## Usage Examples

### Basic UMAP Usage

```python
from src.features.umap import BaseUMAP
import pandas as pd
import numpy as np

# Load your data
data = pd.read_csv('your_data.csv')
labels = np.array([0, 1, 0, 1, ...])  # Class labels

# Initialize UMAP
umap_model = BaseUMAP(
    n_components=2,          # 2D embedding
    n_neighbors=15,          # Local neighborhood size
    min_dist=0.1,           # Minimum distance in embedding
    metric='euclidean',      # Distance metric
    random_state=42         # For reproducibility
)

# Fit and transform
umap_model.fit(data, labels)
transformed_data = umap_model.transform(data)

print(f"Original shape: {data.shape}")
print(f"Embedded shape: {transformed_data.shape}")
```

### High-Dimensional Embedding

```python
# Create higher-dimensional embedding for downstream tasks
umap_hd = BaseUMAP(
    n_components=10,         # 10D embedding
    n_neighbors=30,         # Larger neighborhood
    min_dist=0.0,           # Allow points to be closer
    pca_preprocess=True,    # Use PCA preprocessing
    pca_components=0.95     # Keep 95% variance
)

embedding = umap_hd.fit_transform(data, labels)
```

### Supervised UMAP

```python
# Use labels to guide the embedding
umap_supervised = BaseUMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    supervised=True,         # Enable supervised mode
    random_state=42
)

# Labels are used during fitting to create better separations
supervised_embedding = umap_supervised.fit_transform(data, labels)

# Visualize supervised embedding
umap_supervised.plot_2D_feature_space(
    transformed_data=supervised_embedding,
    labels=labels,
    title="Supervised UMAP Embedding"
)
```

### Transform New Data

```python
# Unlike t-SNE, UMAP can transform new data
new_data = pd.read_csv('new_test_data.csv')

# Transform new data using fitted model
new_embedding = umap_model.transform(new_data)
print(f"New data embedding shape: {new_embedding.shape}")
```

### Parameter Analysis

```python
# Analyze different parameter combinations
umap_model.plot_parameter_analysis(
    data=data,
    labels=labels,
    n_neighbors_range=[5, 15, 30, 50],
    min_dist_range=[0.01, 0.1, 0.5, 0.9],
    save_path="umap_parameter_analysis.png"
)

# Compare different distance metrics
umap_model.plot_metric_comparison(
    data=data,
    labels=labels,
    metrics=['euclidean', 'manhattan', 'cosine'],
    save_path="umap_metrics.png"
)
```

### Advanced Configuration

```python
# High-quality embedding with custom parameters
umap_advanced = BaseUMAP(
    n_components=2,
    n_neighbors=30,          # Larger neighborhood for more global structure
    min_dist=0.01,          # Very tight clusters
    metric='cosine',         # Cosine distance for text/sparse data
    learning_rate=1.5,      # Faster learning
    n_epochs=500,           # More training epochs
    pca_preprocess=True,    # PCA preprocessing
    pca_components=50,      # Fixed number of PCA components
    random_state=42
)

advanced_embedding = umap_advanced.fit_transform(data, labels)
```

### Working with Graph Properties

```python
# Get information about the learned graph structure
graph_props = umap_model.get_graph_properties()
print(f"Graph properties: {graph_props}")

# Get feature and component names
feature_names = umap_model.get_feature_names()
component_names = umap_model.get_component_names()
print(f"Input features: {feature_names}")
print(f"UMAP components: {component_names}")
```

## Parameters

### BaseUMAP Constructor
- `n_components`: Number of embedding dimensions (int, default: 2)
- `n_neighbors`: Local neighborhood size (int, default: 15, range: 2-100)
- `min_dist`: Minimum distance between points in embedding (float, default: 0.1, range: 0.0-1.0)
- `metric`: Distance metric ('euclidean', 'manhattan', 'cosine', etc., default: 'euclidean')
- `learning_rate`: Optimization learning rate (float, default: 1.0)
- `n_epochs`: Training epochs (int, None for auto)
- `pca_preprocess`: Whether to apply PCA preprocessing (bool, default: True)
- `pca_components`: PCA components to keep (int or float, default: 0.9)
- `supervised`: Enable supervised learning (bool, default: False)
- `random_state`: Random seed (int, default: 42)

### Key Parameters Explained

#### n_neighbors
- **Low values (2-10)**: Focus on very local structure, more detailed clusters
- **Medium values (10-30)**: Balanced local/global structure, good default
- **High values (30-100)**: More global structure, smoother embedding

#### min_dist
- **Very low (0.0-0.01)**: Points can be very close, tight clusters
- **Low (0.01-0.1)**: Reasonable separation, good for most cases
- **Medium (0.1-0.5)**: More spread out embedding
- **High (0.5-1.0)**: Very spread out, may lose cluster structure

#### Metrics
- **euclidean**: Standard distance, good for most numerical data
- **manhattan**: L1 distance, robust to outliers
- **cosine**: Good for high-dimensional, sparse data
- **correlation**: For data where correlation matters more than magnitude

## Methods

### Core Methods
- `fit(data, labels)`: Fit UMAP model to data
- `transform(data, labels)`: Transform data using fitted model
- `fit_transform(data, labels)`: Fit and transform in one step

### Analysis Methods
- `get_feature_names()`: Get names of input features used
- `get_component_names()`: Get names of UMAP embedding dimensions
- `get_graph_properties()`: Get properties of the learned neighborhood graph

### Visualization Methods
- `plot_parameter_analysis()`: Compare embeddings with different parameter combinations
- `plot_metric_comparison()`: Compare different distance metrics
- Inherited plotting methods from base class for visualization

## Features

### Advantages Over t-SNE
- **New Data Transform**: Can transform new data points
- **Better Global Structure**: Preserves both local and global relationships
- **Faster Computation**: More efficient for large datasets
- **Flexible Dimensions**: Works well with any number of output dimensions
- **Deterministic**: More stable results across runs

### Advanced Features
- **Supervised Learning**: Use labels to guide embedding creation
- **PCA Preprocessing**: Optional dimensionality reduction before UMAP
- **Multiple Metrics**: Support for various distance measures
- **Parameter Analysis**: Built-in tools for parameter optimization
- **Graph Analysis**: Access to learned neighborhood graph properties

### Preprocessing Options
- **Automatic Standardization**: Built-in data standardization
- **PCA Integration**: Optional PCA for noise reduction and speed
- **Flexible Input**: Works with various data types and shapes

## Dependencies

- umap-learn: UMAP implementation
- scikit-learn: Preprocessing and PCA
- pandas: Data manipulation
- numpy: Numerical computations
- matplotlib: Visualization
- Base module: Inherits from DimensionalityReduction abstract class

## Best Practices

### Parameter Selection
- Start with default parameters (n_neighbors=15, min_dist=0.1)
- Use parameter analysis tools to optimize
- Consider data size when choosing n_neighbors
- Adjust min_dist based on desired cluster tightness

### Data Preparation
- UMAP handles standardization automatically
- Consider PCA preprocessing for very high-dimensional data
- Use appropriate distance metric for your data type

### Interpretation
- Global structure is more meaningful than in t-SNE
- Distances between clusters have more meaning
- Both local neighborhoods and global layout are informative

## Typical Workflow

1. **Data Loading**: Load and prepare your dataset
2. **Initial Embedding**: Create embedding with default parameters
3. **Parameter Tuning**: Use analysis tools to optimize parameters
4. **Final Embedding**: Generate optimized embedding
5. **Visualization**: Create plots to explore the embedding
6. **New Data**: Transform new data points using fitted model
7. **Analysis**: Interpret clusters and patterns in embedding space
