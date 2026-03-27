# t-SNE (t-Distributed Stochastic Neighbor Embedding) Module

## Overview

The `tsne.py` module implements t-SNE for dimensionality reduction and visualization. t-SNE is particularly effective for visualizing high-dimensional data in 2D or 3D space by preserving local neighborhood structures.

## Purpose

- Reduce high-dimensional data to 2D or 3D for visualization
- Preserve local neighborhood relationships between data points
- Reveal clusters and patterns in high-dimensional data
- Provide non-linear dimensionality reduction capabilities
- Optimize visualization quality through parameter tuning

## Main Classes

### BaseTSNE

t-SNE implementation designed for data visualization and exploration.

#### Key Features
- Configurable embedding dimensions (typically 2D or 3D)
- Adjustable perplexity for local/global structure balance
- Multiple initialization methods (PCA, random)
- Built-in parameter analysis tools
- Automatic data standardization

## Usage Examples

### Basic t-SNE Usage

```python
from src.features.tsne import BaseTSNE
import pandas as pd
import numpy as np

# Load your data
data = pd.read_csv('high_dimensional_data.csv')
labels = np.array([0, 1, 0, 1, ...])  # Class labels

# Initialize t-SNE
tsne = BaseTSNE(
    n_components=2,        # 2D embedding
    perplexity=30.0,      # Balance local/global structure
    learning_rate='auto',  # Automatic learning rate
    max_iter=1000,        # Maximum iterations
    random_state=42       # For reproducibility
)

# Fit and transform (t-SNE does both simultaneously)
transformed_data = tsne.fit_transform(data, labels)

print(f"Original shape: {data.shape}")
print(f"Embedded shape: {transformed_data.shape}")
```

### 3D Embedding

```python
# Create 3D t-SNE embedding
tsne_3d = BaseTSNE(
    n_components=3,
    perplexity=50.0,
    max_iter=1500
)

embedding_3d = tsne_3d.fit_transform(data, labels)

# Visualize 3D embedding
tsne_3d.plot_feature_space(
    transformed_data=embedding_3d,
    labels=labels,
    title="3D t-SNE Embedding"
)
```

### Parameter Optimization

```python
# Analyze different perplexity values
tsne.plot_perplexity_analysis(
    data=data,
    labels=labels,
    perplexity_range=[5, 10, 20, 30, 50, 100],
    save_path="perplexity_analysis.png"
)

# Check optimization quality
kl_divergence = tsne.get_kl_divergence()
print(f"Final KL divergence: {kl_divergence}")
```

### Advanced Configuration

```python
# High-quality embedding with custom parameters
tsne_hq = BaseTSNE(
    n_components=2,
    perplexity=40.0,          # Higher for more global structure
    learning_rate=200.0,      # Custom learning rate
    max_iter=2000,           # More iterations for convergence
    init='pca',              # PCA initialization
    random_state=42
)

# Fit and visualize
embedding = tsne_hq.fit_transform(data, labels)

# Visualize with custom styling
tsne_hq.plot_2D_feature_space(
    transformed_data=embedding,
    labels=labels,
    title="High-Quality t-SNE Embedding",
    save_path="tsne_embedding.png"
)
```

### Working with Pre-fitted Model

```python
# Note: t-SNE cannot transform new data
# You must use the same data for fitting and transforming

# Fit the model
tsne.fit(data, labels)

# Transform (only works with same data)
embedding = tsne.transform(data, labels)

# Get model information
feature_names = tsne.get_feature_names()
component_names = tsne.get_component_names()
print(f"Features used: {feature_names}")
print(f"Components: {component_names}")
```

### Quality Assessment

```python
# Assess embedding quality
kl_div = tsne.get_kl_divergence()
print(f"KL Divergence: {kl_div:.4f}")

# Lower KL divergence indicates better preservation of local structure
if kl_div < 1.0:
    print("Good embedding quality")
elif kl_div < 3.0:
    print("Reasonable embedding quality")
else:
    print("Consider adjusting parameters")
```

## Parameters

### BaseTSNE Constructor
- `n_components`: Number of embedding dimensions (int, typically 2 or 3)
- `perplexity`: Controls local vs global structure balance (float, default: 30.0)
- `learning_rate`: Optimization learning rate (float or 'auto', default: 'auto')
- `max_iter`: Maximum optimization iterations (int, default: 1000)
- `init`: Initialization method ('pca' or 'random', default: 'pca')
- `random_state`: Random seed for reproducibility (int, default: 42)

### Key Parameters Explained

#### Perplexity
- **Low values (5-15)**: Focus on very local structure, may create many small clusters
- **Medium values (20-50)**: Balanced local/global structure, good default range
- **High values (50-100)**: More global structure, fewer but larger clusters
- **Rule of thumb**: Should be less than number of data points

#### Learning Rate
- **'auto'**: Automatically set based on data size (recommended)
- **Low values (10-100)**: Slower convergence, may get stuck in local minima
- **High values (200-1000)**: Faster convergence, but may overshoot optimal solution

## Methods

### Core Methods
- `fit(data, labels)`: Fit t-SNE model (stores embedding internally)
- `transform(data, labels)`: Return stored embedding (same data only)
- `fit_transform(data, labels)`: Fit and return embedding in one step

### Analysis Methods
- `get_kl_divergence()`: Get final Kullback-Leibler divergence
- `get_feature_names()`: Get names of input features
- `get_component_names()`: Get names of embedding dimensions

### Visualization Methods
- `plot_perplexity_analysis()`: Compare embeddings with different perplexity values
- Inherited plotting methods from base class for 2D/3D visualization

## Important Notes

### Limitations
- **No New Data Transform**: t-SNE cannot transform new data points
- **Stochastic**: Results may vary between runs (use random_state for reproducibility)
- **Computational Cost**: Can be slow for large datasets (>10,000 points)
- **Parameter Sensitive**: Results highly dependent on perplexity and other parameters

### Best Practices
- **Perplexity Selection**: Try multiple values, typically 5-50
- **Standardization**: Data is automatically standardized
- **Interpretation**: Focus on local clusters, not global distances
- **Multiple Runs**: Run several times with different random states
- **Parameter Tuning**: Use perplexity analysis for optimal parameters

## Features

- **Non-linear Dimensionality Reduction**: Captures complex manifold structures
- **Cluster Revelation**: Excellent for revealing hidden clusters
- **Automatic Standardization**: Built-in data preprocessing
- **Parameter Analysis Tools**: Built-in perplexity analysis
- **Quality Metrics**: KL divergence for embedding quality assessment
- **Flexible Dimensions**: Support for 2D and 3D embeddings
- **Reproducible Results**: Random state control for consistent results

## Dependencies

- scikit-learn: t-SNE implementation and preprocessing
- pandas: Data manipulation
- numpy: Numerical computations
- matplotlib: Visualization
- Base module: Inherits from DimensionalityReduction abstract class

## Typical Workflow

1. **Data Preparation**: Load and preprocess high-dimensional data
2. **Parameter Selection**: Use perplexity analysis to find optimal parameters
3. **Embedding Creation**: Generate t-SNE embedding with chosen parameters
4. **Quality Check**: Assess embedding quality using KL divergence
5. **Visualization**: Create plots to explore discovered patterns
6. **Interpretation**: Analyze clusters and patterns in the embedding space
