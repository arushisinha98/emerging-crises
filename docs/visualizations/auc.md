# AUC Visualization Documentation

The AUC visualization functions in `src/visualizations/auc.py` handle
- ROC curve plotting for model comparison
- AUC-ROC scatter plot analysis for dimensionality reduction experiments
- Multi-model performance visualization

The utilities are designed to work with sklearn models and prediction results provided within the DataFrame, supporting both single and multi-model comparison scenarios.

## Usage Instructions

```python
from src.visualizations.auc import (
    plot_roc_curves,
    plot_auc_roc_scatter
)

# Plot ROC curves for model comparison
results_df = pd.DataFrame({'Model1': y_pred_proba1, 'Model2': y_pred_proba2}, index=y_true)
plot_roc_curves(results_df, title='Model Comparison')

# Plot dimensionality reduction results
auc_data = {'PCA [Linear]': {'Developed, Developed': 0.85, 'Emerging, Emerging': 0.78}}
plot_auc_roc_scatter(auc_data, figsize=(10, 6))
```

## Features

1. **Multi-Model ROC Curves**: Plot and compare ROC curves for multiple models on the same axes
2. **Customizable Styling**: Custom line styles, colors, and plot formatting
3. **Dimensionality Analysis**: Scatter plots for analyzing dimensionality reduction experiments
4. **Group Visualization**: Grouping and labeling of related experiments
5. **Statistical Metrics**: Calculation and display of AUC values

## API Reference

### ROC Curve Functions

#### `plot_roc_curves(results_df, linestyle_map=None, title='AUC-ROC Curves', figsize=(8, 6), label=None)`

Plot multiple ROC curves on the same plot for model comparison.

**Parameters:**
- `results_df` (pd.DataFrame): DataFrame where index contains y_true values and each column contains y_pred_proba for different models
- `linestyle_map` (dict, optional): Map of column names to matplotlib linestyle strings (e.g., {'model1': '-b', 'model2': '--r'})
- `title` (str, optional): Plot title. Defaults to 'AUC-ROC Curves'
- `figsize` (tuple, optional): Figure size as (width, height). Defaults to (8, 6)
- `label` (str, optional): Text label to display in the top left corner of the plot

**Returns:**
- None (displays plot)

**Example:**
```python
import pandas as pd
import numpy as np

# Create sample prediction data
y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
y_pred_model1 = np.array([0.1, 0.3, 0.8, 0.9, 0.2, 0.7, 0.6, 0.4])
y_pred_model2 = np.array([0.2, 0.1, 0.9, 0.8, 0.3, 0.8, 0.7, 0.2])

# Create results DataFrame
results_df = pd.DataFrame({
    'Random Forest': y_pred_model1,
    'Logistic Regression': y_pred_model2
}, index=y_true)

# Plot ROC curves with custom styling
linestyle_map = {
    'Random Forest': '-b',      # Solid blue line
    'Logistic Regression': '--r' # Dashed red line
}

plot_roc_curves(results_df, 
                linestyle_map=linestyle_map,
                title='Financial Crisis Prediction Models',
                figsize=(10, 8),
                label='(a)')
```

**Features:**
- **Automatic AUC Calculation**: Calculates and displays AUC values for each model in the legend
- **Default Styling**: Provides default colors and line styles if none specified
- **Random Guess Baseline**: Includes random guess line for reference
- **Customizable Labels**: Support for subplot labels and custom positioning

**Default Color/Style Cycling:**
- Colors: Blue, Red, Green, Magenta, Cyan, Yellow, Black
- Line styles: Solid, Dashed, Dash-dot, Dotted
- Automatically cycles through combinations for multiple models

### Dimensionality Reduction Analysis Functions

#### `plot_auc_roc_scatter(data, figsize=(10, 6))`

Plot dimensionality reduction AUC-ROC results with separate subplots for each method.

**Parameters:**
- `data` (dict): Dictionary containing AUC-ROC results for different methods and experiments. Values can be None to skip connecting lines
- `figsize` (tuple, optional): Figure size (width, height). Defaults to (10, 6)

**Returns:**
- None (displays plot)

**Raises:**
- `KeyError`: If data dictionary structure is invalid
- `ValueError`: If experiment naming format is incorrect

**Example:**
```python
# Define AUC-ROC results for dimensionality reduction experiments
auc_data = {
    'PCA [Linear]': {
        'Developed, Developed': 0.852,
        'Emerging, Emerging': 0.783,
        'Developed+Emerging, Developed': 0.834,
        'Developed+Emerging, Emerging': 0.756
    },
    'Autoencoder [Non-linear]': {
        'Developed, Developed': 0.867,
        'Emerging, Emerging': None,  # Missing data
        'Developed+Emerging, Developed': 0.845,
        'Developed+Emerging, Emerging': 0.772
    },
    'UMAP [Non-linear]': {
        'Developed, Developed': 0.823,
        'Emerging, Emerging': 0.798,
        'Developed+Emerging, Developed': 0.819,
        'Developed+Emerging, Emerging': 0.784
    }
}

plot_auc_roc_scatter(auc_data, figsize=(12, 8))
```

**Data Structure Requirements:**
The data dictionary should follow this format:
```python
{
    'Method Name [Group]': {
        'Train Data, Test Data': auc_score,
        'Train Data, Test Data': None  # For missing values
    }
}
```

**Experiment Naming Convention:**
- Format: `'Training Market, Testing Market'`
- Training options: `'Developed'`, `'Emerging'`, `'Developed+Emerging'`
- Testing options: `'Developed'`, `'Emerging'`

**Features:**
- **Grouped Methods**: Automatically groups methods by bracketed text (e.g., [Linear], [Non-linear])
- **Missing Data Handling**: Handles None values without breaking connecting lines
- **Market Comparison**: Distinguishes between different training/testing market combinations
- **Visual Markers**: Different markers for different training data configurations
- **Connecting Lines**: Draws lines between related experiments when both values exist

**Marker Legend:**
- **Circle (○)**: Trained on Developed Markets Only
- **Triangle (△)**: Trained on Emerging Markets Only  
- **Diamond (◆)**: Trained on Developed and Emerging Markets

**Subplot Layout:**
- Each dimensionality reduction method gets its own subplot
- Y-axis shows test market (Developed/Emerging)
- X-axis shows AUC-ROC performance
- Connecting lines show performance difference between training configurations

## Visualization Customization

### Color and Style Guidelines

The module follows consistent visual design principles:

```python
# Standard color scheme
marker_color = '#1f77b4'  # Blue for consistency
line_color = 'black'      # Black connecting lines
grid_alpha = 0.3          # Light grid lines
```

### Layout Standards

```python
# Figure sizing recommendations
single_plot = (8, 6)      # Standard single plot
multi_subplot = (10, 6)   # Multi-subplot layouts
comparison = (12, 8)      # Model comparison plots
```

### Text and Label Formatting

```python
# Font size standards
title_size = 14           # Main titles
axis_label_size = 12      # Axis labels  
legend_size = 12          # Legend text
subplot_label_size = 12   # Subplot labels (a), (b), etc.
```

## Best Practices

### 1. Data Preparation

```python
# Ensure results DataFrame has proper structure
results_df = pd.DataFrame({
    'Model_1': y_pred_proba_1,
    'Model_2': y_pred_proba_2
}, index=y_true_labels)

# Verify data types
assert results_df.index.dtype in ['int64', 'bool'], "Index should contain binary labels"
assert all(0 <= results_df.values.flatten()) and all(results_df.values.flatten() <= 1), "Predictions should be probabilities [0,1]"
```

### 2. Model Comparison Workflow

```python
# Standard model comparison pipeline
models = {'Random Forest': rf_model, 'SVM': svm_model, 'Neural Net': nn_model}
results = {}

for name, model in models.items():
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get positive class probabilities
    results[name] = y_pred_proba

results_df = pd.DataFrame(results, index=y_test)
plot_roc_curves(results_df, title='Crisis Prediction Model Comparison')
```

### 3. Dimensionality Reduction Analysis

```python
# Organize results by method and experiment
methods = ['PCA', 'ICA', 'Autoencoder', 'UMAP']
train_test_combinations = [
    ('Developed', 'Developed'),
    ('Emerging', 'Emerging'), 
    ('Developed+Emerging', 'Developed'),
    ('Developed+Emerging', 'Emerging')
]

auc_results = {}
for method in methods:
    method_key = f"{method} [{'Linear' if method in ['PCA', 'ICA'] else 'Non-linear'}]"
    auc_results[method_key] = {}
    
    for train_data, test_data in train_test_combinations:
        experiment_key = f"{train_data}, {test_data}"
        # Run experiment and get AUC score
        auc_score = run_experiment(method, train_data, test_data)
        auc_results[method_key][experiment_key] = auc_score

plot_auc_roc_scatter(auc_results)
```

### 4. Handling Missing Data

```python
# Handle experiments that failed or weren't run
auc_results = {
    'Method 1': {
        'Developed, Developed': 0.85,
        'Emerging, Emerging': None,  # Experiment failed
        'Developed+Emerging, Developed': 0.83,
        'Developed+Emerging, Emerging': 0.78
    }
}

# The visualization will automatically skip None values for connecting lines
plot_auc_roc_scatter(auc_results)
```

## Common Use Cases

### Financial Crisis Prediction Model Evaluation

```python
# Compare different machine learning models
crisis_models = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model, 
    'Support Vector Machine': svm_model,
    'Neural Network': nn_model
}

# Get predictions for test set
results = {}
for name, model in crisis_models.items():
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        # For models without predict_proba, use decision function
        scores = model.decision_function(X_test)
        proba = 1 / (1 + np.exp(-scores))  # Sigmoid transformation
    
    results[name] = proba

results_df = pd.DataFrame(results, index=y_test)

# Create comparison plot
plot_roc_curves(results_df, 
                title='Financial Crisis Prediction Models - ROC Comparison',
                figsize=(10, 8))
```

### Dimensionality Reduction Impact Analysis

```python
# Analyze how different dimensionality reduction techniques affect performance
dimensionality_methods = {
    'None (Original Features)': original_results,
    'PCA [Linear]': pca_results,
    'Factor Analysis [Linear]': fa_results, 
    'Autoencoder [Non-linear]': ae_results,
    'UMAP [Non-linear]': umap_results
}

# Plot comprehensive comparison
plot_auc_roc_scatter(dimensionality_methods, figsize=(12, 10))
```

### Cross-Market Performance Analysis

```python
# Analyze model performance across different economic markets
market_analysis = {}

for method in ['PCA', 'Autoencoder', 'UMAP']:
    group = 'Linear' if method == 'PCA' else 'Non-linear'
    method_key = f"{method} [{group}]"
    
    market_analysis[method_key] = {
        'Developed, Developed': train_test_model(method, 'developed', 'developed'),
        'Emerging, Emerging': train_test_model(method, 'emerging', 'emerging'),
        'Developed+Emerging, Developed': train_test_model(method, 'mixed', 'developed'),
        'Developed+Emerging, Emerging': train_test_model(method, 'mixed', 'emerging')
    }

plot_auc_roc_scatter(market_analysis, figsize=(14, 8))
```

## Technical Implementation Details

### ROC Curve Calculation

The module uses scikit-learn's `roc_curve` and `auc` functions:

```python
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve points
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot with AUC in legend
plt.plot(fpr, tpr, label=f'Model (AUC = {roc_auc:.3f})')
```

### Subplot Management

For dimensionality reduction plots, the module dynamically creates subplot layouts:

```python
# Dynamic subplot creation based on number of methods
n_methods = len(ordered_methods)
fig, axes = plt.subplots(n_methods, 1, figsize=figsize, sharex=True)

# Handle single subplot case
if n_methods == 1:
    axes = [axes]
```

### Group Detection Algorithm

Method grouping is automatically detected from bracketed text:

```python
# Parse method names for grouping
if '[' in method_name and ']' in method_name:
    start_bracket = method_name.find('[')
    end_bracket = method_name.find(']')
    group_text = method_name[start_bracket+1:end_bracket]
    clean_name = method_name[:start_bracket].strip()
```
