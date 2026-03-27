# Model Evaluation and Classification Utilities

## Overview

The `classifiers.py` module provides utility functions for model evaluation and analysis. It includes comprehensive plotting functions for model performance visualization, feature importance analysis, and hyperparameter optimization utilities.

## Purpose

- Provide standardized model evaluation visualizations
- Support both traditional ML and deep learning model evaluation
- Enable feature importance analysis with grouping capabilities
- Offer hyperparameter tuning utilities with stratified cross-validation
- Handle sequential model evaluation with proper alignment

## Main Functions

### plot_metrics()

Comprehensive model evaluation with confusion matrix and ROC curve visualization.

### plot_feature_importances()

Advanced feature importance visualization with optional feature grouping and styling.

### stratified_gridsearch()

Stratified grid search with cross-validation for hyperparameter optimization.

## Usage Examples

### Basic Model Evaluation

```python
from src.model.classifiers import plot_metrics
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Train a standard sklearn model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate with comprehensive metrics
plot_metrics(
    model=rf,
    X_test=X_test,
    y_test=y_test
)

# This will display:
# - Confusion matrix (as percentages)
# - ROC curve with AUC score
# - Automatic handling of probability extraction
```

### Deep Learning Model Evaluation

```python
from src.model.architectures import LSTMClassifier

# Train LSTM model (handles sequences automatically)
lstm = LSTMClassifier(params=lstm_params)
lstm.fit(train_data, train_labels)

# Evaluate sequential model (no y_test needed - model handles alignment)
plot_metrics(
    model=lstm,
    X_test=test_sequential_data,
    y_test=test_labels  # Will be aligned automatically
)

# LSTM models provide additional functionality:
# - Automatic sequence alignment
# - Proper handling of temporal dependencies
# - Index-based prediction matching
```

### Feature Importance Analysis

```python
from src.model.classifiers import plot_feature_importances

# Train a tree-based model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Get feature names
feature_names = X_train.columns.tolist()

# Basic feature importance plot
top_features = plot_feature_importances(
    model=rf,
    feature_names=feature_names,
    top_n=25,
    title="Top 25 Important Features"
)

print(f"Most important features: {top_features[:5]}")
```

### Grouped Feature Importance

```python
# Define feature groups for better visualization
feature_groups = {
    'Macroeconomic': [
        'GDP_growth', 'inflation_rate', 'unemployment_rate',
        'interest_rate', 'government_debt_pct_gdp'
    ],
    'Financial Markets': [
        'stock_market_volatility', 'credit_spread', 'currency_volatility',
        'banking_sector_health', 'liquidity_ratio'
    ],
    'Trade & External': [
        'current_account_balance', 'trade_balance', 'foreign_reserves',
        'export_growth', 'import_growth'
    ],
    'Corporate': [
        'corporate_debt_ratio', 'profit_margins', 'investment_rate',
        'business_confidence', 'credit_growth'
    ]
}

# Plot with feature grouping and custom styling
top_features = plot_feature_importances(
    model=rf,
    feature_names=feature_names,
    top_n=30,
    feature_groups=feature_groups,
    title="Feature Importance by Category"
)

# Features will be color-coded by group with legend
# Even groups: filled bars
# Odd groups: hatched bars
```

### Hyperparameter Optimization

```python
from src.model.classifiers import stratified_gridsearch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Run stratified grid search with cross-validation
base_model = RandomForestClassifier(random_state=42)
grid_search = stratified_gridsearch(
    param_grid=param_grid,
    base_model=base_model,
    X_train=X_train,
    y_train=y_train,
    scoring='f1_weighted'  # Good for imbalanced data
)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Get the best model
best_model = grid_search.best_estimator_
```

### Advanced Grid Search Example

```python
from xgboost import XGBClassifier

# XGBoost parameter grid
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Grid search with different scoring metrics
scoring_options = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

results = {}
for scoring in scoring_options:
    grid_search = stratified_gridsearch(
        param_grid=xgb_param_grid,
        base_model=XGBClassifier(random_state=42),
        X_train=X_train,
        y_train=y_train,
        scoring=scoring
    )
    
    results[scoring] = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': grid_search.best_estimator_
    }

# Compare results across different scoring metrics
for metric, result in results.items():
    print(f"{metric}: {result['best_score']:.4f} with {result['best_params']}")
```

### Evaluation Pipeline

```python
def evaluate_model_comprehensive(model, X_test, y_test, feature_names, 
                                feature_groups=None, model_name="Model"):
    """Comprehensive model evaluation pipeline"""
    
    print(f"Evaluating {model_name}...")
    
    # 1. Performance metrics visualization
    plot_metrics(model, X_test, y_test)
    
    # 2. Feature importance analysis (if available)
    if hasattr(model, 'feature_importances_'):
        print(f"\n{model_name} Feature Importance Analysis:")
        top_features = plot_feature_importances(
            model=model,
            feature_names=feature_names,
            top_n=20,
            feature_groups=feature_groups,
            title=f"{model_name} - Top 20 Features"
        )
        return top_features
    else:
        print(f"{model_name} does not support feature importance analysis")
        return None

# Usage
top_rf_features = evaluate_model_comprehensive(
    model=best_rf_model,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_names,
    feature_groups=feature_groups,
    model_name="Random Forest"
)
```

## Parameters

### plot_metrics()
- `model`: Trained model (sklearn-compatible or custom with predict/predict_proba methods)
- `X_test`: Test features (DataFrame for sequential models, array/DataFrame for others)
- `y_test`: True test labels (np.array, optional for sequential models)

### plot_feature_importances()
- `model`: Trained model with feature_importances_ attribute
- `feature_names`: List of feature names (List[str])
- `top_n`: Number of top features to display (int, default: 25)
- `feature_groups`: Dictionary mapping group names to feature lists (dict, optional)
- `title`: Plot title (str, default: "Feature Importances")

### stratified_gridsearch()
- `param_grid`: Parameter grid for search (dict)
- `base_model`: Base model class to optimize
- `X_train`: Training features
- `y_train`: Training labels
- `scoring`: Scoring metric (str, default: 'f1_weighted')

## Function Details

### plot_metrics() Features

#### Model Type Detection
- **Standard Models**: Uses predict() and predict_proba() methods
- **Sequential Models**: Uses predict_with_indices() and get_aligned_labels()
- **Probability Handling**: Automatically extracts positive class probabilities
- **Label Alignment**: Handles temporal sequence alignment for LSTM models

#### Visualizations
- **Confusion Matrix**: Displayed as percentages with color coding
- **ROC Curve**: With AUC score and random baseline comparison
- **Automatic Scaling**: Color limits and axis scaling optimized for clarity

### plot_feature_importances() Features

#### Styling Options
- **Feature Grouping**: Color-coded groups with legend
- **Bar Styling**: Alternating filled and hatched patterns for groups
- **Color Palette**: Automatic color assignment from matplotlib tab20
- **Layout**: Rotated labels and proper spacing

#### Advanced Features
- **Top-N Selection**: Automatically selects most important features
- **Group Legend**: Shows feature categories with matching colors/patterns
- **Return Values**: Returns list of top feature names for further analysis

### stratified_gridsearch() Features

#### Cross-Validation
- **Stratified Folds**: Maintains class distribution across folds
- **Train/Validation Split**: Uses 80/20 split for faster optimization
- **Reproducible**: Fixed random seeds for consistent results
- **Verbose Output**: Progress tracking during search

## Advanced Usage Patterns

### Model Comparison Pipeline

```python
def compare_models(models_dict, X_test, y_test, feature_names):
    """Compare multiple models side by side"""
    
    results = {}
    for model_name, model in models_dict.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print('='*50)
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Store results
        results[model_name] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # Plot metrics
        plot_metrics(model, X_test, y_test)
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            top_features = plot_feature_importances(
                model, feature_names, top_n=15,
                title=f"{model_name} - Feature Importance"
            )
            results[model_name]['top_features'] = top_features
    
    return results

# Usage
models = {
    'Random Forest': best_rf,
    'XGBoost': best_xgb,
    'LSTM': best_lstm
}

comparison_results = compare_models(models, X_test, y_test, feature_names)
```

### Hyperparameter Sensitivity Analysis

```python
def analyze_param_sensitivity(base_model, param_name, param_values, 
                             X_train, y_train, scoring='f1_weighted'):
    """Analyze sensitivity to specific parameter"""
    
    scores = []
    for value in param_values:
        # Create parameter grid with single parameter
        param_grid = {param_name: [value]}
        
        # Run grid search
        grid_search = stratified_gridsearch(
            param_grid, base_model, X_train, y_train, scoring
        )
        
        scores.append(grid_search.best_score_)
    
    # Plot sensitivity
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, scores, 'o-', linewidth=2, markersize=8)
    plt.xlabel(param_name)
    plt.ylabel(f'{scoring.upper()} Score')
    plt.title(f'Parameter Sensitivity: {param_name}')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return dict(zip(param_values, scores))

# Example usage
rf_depths = analyze_param_sensitivity(
    base_model=RandomForestClassifier(n_estimators=100, random_state=42),
    param_name='max_depth',
    param_values=[3, 5, 10, 15, 20, None],
    X_train=X_train,
    y_train=y_train
)
```

## Dependencies

- matplotlib: Plotting and visualization
- numpy: Numerical computations
- pandas: Data manipulation
- scikit-learn: Model evaluation metrics and cross-validation
- seaborn: Enhanced statistical visualizations (optional)

## Best Practices

### Evaluation Guidelines
- Always use both confusion matrix and ROC curve for binary classification
- Consider precision-recall curves for highly imbalanced datasets
- Use stratified cross-validation to maintain class distributions
- Evaluate feature importance to understand model decisions

### Visualization Tips
- Use feature grouping for interpretable importance plots
- Include model name and key parameters in plot titles
- Save plots for documentation and reporting
- Use consistent color schemes across related plots

### Hyperparameter Tuning
- Start with coarse grid search, then refine around best parameters
- Use appropriate scoring metrics for your problem (F1 for imbalanced data)
- Consider computational cost vs. performance gains
- Validate final model on independent test set

## Error Handling

The module handles common issues:
- Models without feature importance attributes
- Missing prediction methods
- Inconsistent probability array shapes
- Empty or invalid parameter grids
- Cross-validation failures

## Integration

Works seamlessly with:
- All sklearn-compatible models
- Custom PyTorch models from architectures.py
- Feature engineering pipelines
- Model selection workflows
- Automated ML pipelines
