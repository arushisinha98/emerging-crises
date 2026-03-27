# Base Dimensionality Reduction Module

## Overview

The `base.py` module provides the abstract base class `DimensionalityReduction` that serves as the foundation for all dimensionality reduction techniques in this project. It defines the common interface and shared functionality for dimensionality reduction methods like PCA, t-SNE, UMAP, and VAE.

## Purpose

- Provides a standardized interface for all dimensionality reduction implementations
- Defines common visualization methods for 1D and 2D feature spaces
- Implements utility functions for data preparation and component analysis
- Ensures consistent behavior across different dimensionality reduction techniques

## Main Classes

### DimensionalityReduction (Abstract Base Class)

An abstract base class that defines the interface for dimensionality reduction implementations.

#### Key Methods

- `fit(data, labels)`: Fit the model to the provided data
- `transform(data, labels)`: Transform data using the fitted model
- `plot_1D_feature_space()`: Visualize 1D feature space using violin plots
- `plot_2D_feature_space()`: Visualize 2D feature space with scatter plots
- `plot_feature_space()`: General feature space visualization method

#### Utility Methods

- `to_dataframe()`: Convert transformed data to DataFrame format
- `label_to_colors()`: Convert labels to color mappings for visualization
- `choose_components()`: Select components based on spread metrics
- `choose_categories()`: Select categories for visualization using distance metrics

## Usage Examples

### Basic Implementation Pattern

```python
from src.features.base import DimensionalityReduction
import pandas as pd
import numpy as np

class MyDimReduction(DimensionalityReduction):
    def fit(self, data: pd.DataFrame, labels: np.array):
        # Implementation specific fitting logic
        return self
    
    def transform(self, data: pd.DataFrame, labels: np.array = None):
        # Implementation specific transformation logic
        return transformed_data

# Usage
model = MyDimReduction(n_components=2)
model.fit(data, labels)
transformed = model.transform(data)
```

### Visualization Example

```python
# Visualize 2D feature space
model.plot_2D_feature_space(
    transformed_data=transformed,
    labels=labels,
    title="My Dimensionality Reduction",
    save_path="output/visualization.png"
)

# Visualize 1D feature space
model.plot_1D_feature_space(
    transformed_data=transformed,
    labels=labels,
    idx=0,  # First component
    title="First Component Distribution"
)
```

### Component Selection

```python
from src.features.base import choose_components

# Choose best component pair based on determinant spread
comp1, comp2 = choose_components(
    transformed_data, 
    method='determinant'
)
print(f"Best components: {comp1}, {comp2}")
```

## Parameters

### DimensionalityReduction Constructor

- `n_components`: Number of components to keep (int) or variance threshold (float, 0-1)

### Visualization Methods

- `transformed_data`: DataFrame with transformed data
- `labels`: Array of class labels for coloring points
- `title`: Plot title
- `save_path`: Optional path to save the plot
- `show`: Whether to display the plot (default: True)
- `color_by`: Categories for coloring points (optional)

## Features

- **Abstract Interface**: Ensures consistent API across all dimensionality reduction methods
- **Rich Visualizations**: Built-in plotting methods for exploring transformed data
- **Flexible Component Selection**: Multiple metrics for choosing optimal components
- **Color-coded Visualizations**: Automatic color mapping for different categories
- **Metadata Preservation**: Maintains data structure and indexing through transformations

## Dependencies

- pandas: Data manipulation and DataFrame operations
- numpy: Numerical computations
- matplotlib: Plotting and visualization
- seaborn: Enhanced statistical visualizations
- abc: Abstract base class functionality
