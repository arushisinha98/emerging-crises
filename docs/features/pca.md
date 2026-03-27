# Principal Component Analysis (PCA) Module

## Overview

The `pca.py` module implements Principal Component Analysis (PCA) for dimensionality reduction. It provides two main implementations: `BasePCA` for standard PCA analysis and `ClassSpecificPCA` for class-aware dimensionality reduction.

## Purpose

- Perform linear dimensionality reduction using PCA
- Provide separate PCA models for different classes (majority/minority)
- Extract principal components that capture maximum variance in data
- Analyze feature contributions to each principal component
- Visualize explained variance and component loadings

## Main Classes

### BasePCA

Standard PCA implementation that applies dimensionality reduction row-wise on the data.

#### Key Features
- Automatic data standardization
- Configurable number of components or variance threshold
- Feature importance analysis
- Explained variance visualization

### ClassSpecificPCA

Class-specific PCA that creates separate models for majority and minority classes.

#### Key Features
- Separate PCA models for each class
- Class-aware dimensionality reduction
- Beneficial for imbalanced datasets
- Independent component analysis per class

## Usage Examples

### Basic PCA Usage

```python
from src.features.pca import BasePCA
import pandas as pd
import numpy as np

# Load your data
data = pd.read_csv('your_data.csv')
labels = np.array([0, 1, 0, 1, ...])  # Binary labels

# Initialize PCA with 2 components
pca = BasePCA(n_components=2)

# Fit and transform
pca.fit(data, labels)
transformed_data = pca.transform(data)

print(f"Original shape: {data.shape}")
print(f"Transformed shape: {transformed_data.shape}")
```

### Variance Threshold PCA

```python
# Use variance threshold instead of fixed components
pca = BasePCA(n_components=0.95)  # Keep 95% of variance
pca.fit(data, labels)
transformed_data = pca.transform(data)

# Check how many components were selected
print(f"Components selected: {len(pca.get_component_names())}")
```

### Feature Importance Analysis

```python
# Get top features contributing to first principal component
top_features = pca.get_top_components(PC=1, n=10)
print(f"Top 10 features for PC1: {top_features}")

# Get explained variance ratio
explained_var = pca.get_explained_variance_ratio()
print(f"Explained variance ratios: {explained_var}")
```

### Visualization

```python
# Plot explained variance
pca.plot_explained_variance(
    cumulative=True,
    title="PCA Explained Variance",
    save_path="pca_variance.png"
)

# Visualize 2D feature space
pca.plot_2D_feature_space(
    transformed_data=transformed_data,
    labels=labels,
    title="PCA 2D Projection"
)
```

### Class-Specific PCA

```python
from src.features.pca import ClassSpecificPCA

# Initialize class-specific PCA
class_pca = ClassSpecificPCA(n_components=0.9)

# Fit with labeled data
class_pca.fit(data, labels)

# Transform data (uses appropriate model based on labels)
transformed_data = class_pca.transform(data, labels)

# Get component names for each class
majority_components = class_pca.get_component_names('majority')
minority_components = class_pca.get_component_names('minority')
```

### Advanced Analysis

```python
# Get top-ranking components across both classes (ClassSpecificPCA)
top_rank_features = class_pca.get_top_rank_components(n=40)

# Plot component loadings comparison
class_pca.plot_component_loadings(
    n=30,
    title="Component Loadings Comparison",
    save_path="loadings_comparison.png"
)
```

## Parameters

### BasePCA Constructor
- `n_components`: Number of components (int) or variance threshold (float, 0-1)

### ClassSpecificPCA Constructor
- `n_components`: Number of components (int) or variance threshold (float, 0-1)

### Key Methods Parameters
- `data`: DataFrame with time series data
- `labels`: Array of class labels (0 for majority, 1 for minority)
- `PC`: Principal component index (1-based) for analysis
- `n`: Number of top features to return
- `cumulative`: Whether to plot cumulative variance (default: True)

## Methods

### BasePCA Methods
- `fit(data, labels)`: Fit PCA model to data
- `transform(data, labels)`: Transform data using fitted PCA
- `get_explained_variance_ratio()`: Get variance explained by each component
- `get_top_components(PC, n)`: Get top n features for specified component
- `plot_explained_variance()`: Visualize explained variance
- `get_feature_names()`: Get names of features used in fitting
- `get_component_names()`: Get names of principal components

### ClassSpecificPCA Methods
- `fit(data, labels)`: Fit separate PCA models for each class
- `transform(data, labels)`: Transform using class-appropriate model
- `get_top_components(pc_type, PC, n)`: Get top features for specific class/component
- `get_top_rank_components(n)`: Get top-ranking features across classes
- `plot_component_loadings()`: Compare component loadings between classes

## Features

- **Automatic Standardization**: Data is automatically standardized before PCA
- **Flexible Component Selection**: Support for both fixed components and variance thresholds
- **Class-Aware Analysis**: Separate models for different classes in imbalanced datasets
- **Feature Importance**: Identify which original features contribute most to components
- **Rich Visualizations**: Built-in plotting for explained variance and feature space
- **Robust Validation**: Input validation and error handling

## Dependencies

- scikit-learn: PCA implementation and preprocessing
- pandas: Data manipulation
- numpy: Numerical computations
- matplotlib: Visualization
- Base module: Inherits from DimensionalityReduction abstract class
