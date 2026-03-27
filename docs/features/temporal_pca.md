# Temporal PCA Analysis Module

## Overview

The `temporal_pca.py` module provides specialized PCA implementations for time series data analysis. It includes two main approaches: Sliding Window PCA and Dynamic PCA, both designed to handle temporal patterns while avoiding look-ahead bias.

## Purpose

- Apply PCA to time series data with temporal awareness
- Detect evolving patterns and structural changes over time
- Provide sliding window analysis for local pattern detection
- Implement dynamic PCA with adaptive learning capabilities
- Handle trend and seasonality removal for cleaner analysis

## Main Classes

### TimeSeriesPCABase (Abstract)

Abstract base class providing common functionality for time series PCA implementations.

#### Key Features
- Data preparation without look-ahead bias
- Automatic trend and seasonality removal
- Standardization with temporal awareness
- Common interface for time series PCA methods

### SlidingWindowPCA

Applies PCA to fixed-length windows of time series data, sliding forward step by step.

#### Key Features
- Fixed window size analysis
- Configurable step size for window movement
- Captures gradual changes in data patterns
- Maintains temporal locality
- Identifies structural breaks and regime changes

### DynamicPCA

Implements dynamic PCA with continuous adaptation to new data using recursive updates.

#### Key Features
- Exponential forgetting factor for recent data emphasis
- Recursive PCA updates
- Country-specific processing
- Adaptive learning capabilities
- Maintains relevance to recent patterns

## Usage Examples

### Sliding Window PCA

```python
from src.features.temporal_pca import SlidingWindowPCA
import pandas as pd

# Load time series data
ts_data = pd.read_csv('time_series_data.csv', index_col='date', parse_dates=True)

# Initialize Sliding Window PCA
sw_pca = SlidingWindowPCA(
    name="financial_crisis_analysis",
    window_size=30,      # 30 time periods per window
    step_size=1,         # Move window by 1 period
    n_components=3,      # Keep 3 components
    standardize=True,
    remove_trend=True
)

# Fit the model
sw_pca.fit(ts_data)

# Get transformed data for all windows
transformed_windows = sw_pca.fit_transform(ts_data)

# Each element in the list is a transformed window
print(f"Number of windows: {len(transformed_windows)}")
print(f"Shape per window: {transformed_windows[0].shape}")
```

### Dynamic PCA

```python
from src.features.temporal_pca import DynamicPCA

# Initialize Dynamic PCA
dyn_pca = DynamicPCA(
    name="adaptive_analysis",
    forgetting_factor=0.95,    # Exponential forgetting
    min_samples=10,            # Minimum samples before updates
    update_frequency=1,        # Update every period
    n_components=0.9,         # 90% variance
    remove_seasonality=True,
    seasonality_period=12
)

# Fit and transform
dyn_pca.fit(ts_data)
transformed_data = dyn_pca.fit_transform(ts_data)

# transformed_data is a dict with country-specific results
for country, data in transformed_data.items():
    print(f"{country}: {data.shape}")
```

### Advanced Configuration

```python
# Sliding Window PCA with custom preprocessing
sw_pca = SlidingWindowPCA(
    name="custom_analysis",
    window_size=60,
    step_size=5,               # Skip 5 periods between windows
    n_components=0.95,         # Variance threshold
    standardize=True,
    remove_trend=True,
    remove_seasonality=True,
    trend_method='rolling_mean',
    seasonality_period=12
)

# Fit and analyze
sw_pca.fit(ts_data)
transformed = sw_pca.fit_transform(ts_data)
```

### Visualization

```python
# Plot scree plot (explained variance)
sw_pca.plot_scree_plot(
    aggregate_by="last",           # Use last window for aggregation
    countries=["USA", "GBR"],      # Specific countries
    save_path="scree_plot.png"
)

# Plot component loadings evolution
sw_pca.plot_component_loadings(
    n_components=3,
    color_by="max",               # Color by maximum loading
    countries=["USA"],
    save_path="loadings.png"
)

# Plot evolving variance with crisis labels
sw_pca.plot_evolving_variance(
    save_path="variance_evolution.png",
    label_file="crisis_labels.csv"
)

# Plot eigenvalue features
sw_pca.plot_eigenvalue_features(
    save_path="eigenvalue_features.png",
    label_file="crisis_labels.csv"
)
```

### Dynamic PCA Specific Visualizations

```python
# Dynamic PCA visualizations
dyn_pca.plot_evolving_variance(
    save_path="dynamic_variance.png",
    label_file="crisis_labels.csv"
)

# Component loadings for dynamic PCA
dyn_pca.plot_component_loadings(
    n_components=2,
    countries=["USA", "GBR", "JPN"],
    save_path="dynamic_loadings.png"
)
```

## Parameters

### SlidingWindowPCA Constructor
- `name`: Identifier for the analysis
- `window_size`: Size of each analysis window
- `step_size`: Step size for window movement
- `n_components`: Number of components or variance threshold
- `standardize`: Whether to standardize data (default: True)
- `remove_trend`: Whether to remove trends (default: True)
- `remove_seasonality`: Whether to remove seasonality (default: False)
- `trend_method`: Method for trend removal ('rolling_mean', etc.)
- `seasonality_period`: Period for seasonality removal (default: 12)

### DynamicPCA Constructor
- `name`: Identifier for the analysis
- `forgetting_factor`: Weight for exponential forgetting (default: 0.95)
- `min_samples`: Minimum samples before updates (default: 10)
- `update_frequency`: How often to update the model (default: 1)
- Additional parameters inherited from TimeSeriesPCABase

## Methods

### Common Methods (Both Classes)
- `fit(data)`: Fit the temporal PCA model
- `fit_transform(data)`: Fit and transform in one step
- `plot_scree_plot()`: Visualize explained variance over time
- `plot_component_loadings()`: Show component loadings evolution
- `plot_evolving_variance()`: Plot variance evolution with crisis indicators

### SlidingWindowPCA Specific
- `transform(data, n_components)`: Transform data using fitted model
- Returns list of transformed windows

### DynamicPCA Specific
- `fit_transform(data)`: Returns dictionary with country-specific transformations
- Recursive updates based on forgetting factor

## Features

### Data Preprocessing
- **No Look-ahead Bias**: Ensures temporal integrity in analysis
- **Automatic Standardization**: Per-window standardization for sliding window
- **Trend Removal**: Multiple methods for detrending data
- **Seasonality Handling**: Optional seasonal pattern removal

### Analysis Capabilities
- **Temporal Pattern Detection**: Identify evolving patterns over time
- **Structural Break Detection**: Spot regime changes and shifts
- **Country-specific Analysis**: Independent processing per country
- **Adaptive Learning**: Dynamic adjustment to new patterns

### Visualizations
- **Evolving Variance Plots**: Show how explained variance changes over time
- **Component Loading Evolution**: Track how feature importance changes
- **Crisis Overlay**: Visualize analysis results with crisis period indicators
- **Eigenvalue Features**: Advanced eigenvalue-based analysis

## Dependencies

- pandas: Time series data manipulation
- numpy: Numerical computations
- scikit-learn: Preprocessing utilities
- scipy: Statistical functions
- matplotlib: Visualization
- abc: Abstract base class functionality
