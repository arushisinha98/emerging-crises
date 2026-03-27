# Rolling Window Model Framework

## Overview

The `rolling_model.py` module implements a sophisticated time-series cross-validation framework for financial crisis prediction. It provides rolling window training and testing to ensure no data leakage while maintaining temporal order in model evaluation.

## Purpose

- Implement time-series cross-validation with strict temporal ordering
- Prevent data leakage in financial crisis prediction models
- Provide rolling window training with incremental data updates
- Support comprehensive evaluation across multiple time periods
- Enable realistic performance assessment for deployment scenarios

## Main Classes

### RollingWindowModel

Core class implementing the rolling window training and evaluation framework.

#### Key Features
- Temporal order preservation to prevent look-ahead bias
- Configurable test and retrain window sizes
- Automatic feature engineering and label generation
- Comprehensive metrics tracking across all windows
- Feature importance analysis over time

### RollingWindowModelAdapter

Adapter class providing sklearn-compatible interface for plotting and evaluation functions.

#### Key Features
- sklearn-style predict and predict_proba methods
- Seamless integration with existing evaluation pipelines
- Compatible with plotting utilities
- Wraps RollingWindowModel for external use

## Usage Examples

### Basic Rolling Window Evaluation

```python
from src.model.rolling_model import RollingWindowModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Load your data
train_df = pd.read_csv('historical_data.csv')  # Historical training data
test_df = pd.read_csv('test_period_data.csv')  # Data for rolling evaluation

# Pre-computed labels (optional)
train_labels = np.array([0, 1, 0, 1, ...])
test_labels = np.array([0, 1, 0, 1, ...])

# Define model parameters
rf_params = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

# Initialize rolling window model
rolling_model = RollingWindowModel(
    model_class=RandomForestClassifier,
    model_params=rf_params,
    test_window_months=12,      # Test on 12 months at a time
    retrain_window_months=12    # Add 12 months to training data each iteration
)

# Run rolling window evaluation
results = rolling_model.fit_predict_rolling(
    train_df=train_df,
    test_df=test_df,
    train_labels=train_labels,  # Optional: will build labels if None
    test_labels=test_labels,    # Optional: will build labels if None
    verbose=True               # Show progress information
)

# View results
print("Overall Performance:")
for metric, value in results['overall_metrics'].items():
    print(f"  {metric}: {value:.4f}")
```

### Advanced Rolling Window Configuration

```python
from xgboost import XGBClassifier

# XGBoost with custom parameters
xgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'logloss'
}

# More frequent retraining
rolling_xgb = RollingWindowModel(
    model_class=XGBClassifier,
    model_params=xgb_params,
    test_window_months=6,       # Test on 6 months
    retrain_window_months=6     # Retrain every 6 months
)

# Run evaluation
xgb_results = rolling_xgb.fit_predict_rolling(
    train_df=train_df,
    test_df=test_df,
    verbose=True
)

# Analyze window-specific performance
for i, window_result in enumerate(xgb_results['window_results']):
    print(f"Window {i+1}: F1={window_result['f1']:.3f}, "
          f"AUC={window_result['auc']:.3f}, "
          f"Samples={window_result['n_samples']}")
```

### Feature Importance Over Time

```python
# Analyze feature importance evolution
top_features = rolling_model.get_feature_importances(
    model_idx=-1,  # Use last trained model
    top_n=20      # Top 20 features
)

print("Top Features (Latest Model):")
for feature, importance in top_features:
    print(f"  {feature}: {importance:.4f}")

# Plot feature importance history
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8))
importance_history = results['feature_importances_history']

# Plot evolution of top 5 features
top_5_features = [feat[0] for feat in top_features[:5]]
for feature in top_5_features:
    importances = [window.get(feature, 0) for window in importance_history]
    ax.plot(range(len(importances)), importances, marker='o', label=feature)

ax.set_xlabel('Window Number')
ax.set_ylabel('Feature Importance')
ax.set_title('Feature Importance Evolution Over Time')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### Model Adapter for Evaluation

```python
from src.model.rolling_model import RollingWindowModelAdapter
from src.model.classifiers import plot_metrics

# Create adapter for sklearn-compatible interface
adapter = RollingWindowModelAdapter(rolling_model)

# Use with existing evaluation functions
# Note: adapter handles prediction alignment automatically
plot_metrics(
    model=adapter,
    X_test=test_df,      # Full test DataFrame
    y_test=None         # Not needed - adapter handles this
)

# Get predictions for custom analysis
predictions_df = adapter.get_predictions_for_plotting()
print("Predictions DataFrame columns:", predictions_df.columns.tolist())
print("Predictions shape:", predictions_df.shape)
```

### Custom Evaluation Workflow

```python
def comprehensive_rolling_evaluation(model_class, model_params, 
                                   train_df, test_df, test_windows=[6, 12, 24]):
    """Evaluate model with different window sizes"""
    
    results = {}
    
    for window_size in test_windows:
        print(f"\nEvaluating with {window_size}-month windows...")
        
        rolling_model = RollingWindowModel(
            model_class=model_class,
            model_params=model_params,
            test_window_months=window_size,
            retrain_window_months=window_size
        )
        
        window_results = rolling_model.fit_predict_rolling(
            train_df=train_df,
            test_df=test_df,
            verbose=False
        )
        
        results[f'{window_size}_months'] = {
            'overall_f1': window_results['overall_metrics']['overall_f1'],
            'overall_precision': window_results['overall_metrics']['overall_precision'],
            'overall_recall': window_results['overall_metrics']['overall_recall'],
            'n_windows': len(window_results['window_results'])
        }
    
    return results

# Compare different window sizes
from sklearn.ensemble import RandomForestClassifier

comparison_results = comprehensive_rolling_evaluation(
    model_class=RandomForestClassifier,
    model_params={'n_estimators': 200, 'random_state': 42},
    train_df=train_df,
    test_df=test_df,
    test_windows=[6, 12, 18, 24]
)

# Print comparison
print("\nWindow Size Comparison:")
for window, metrics in comparison_results.items():
    print(f"{window}: F1={metrics['overall_f1']:.3f}, "
          f"Precision={metrics['overall_precision']:.3f}, "
          f"Recall={metrics['overall_recall']:.3f}")
```

### Time Series Validation Pipeline

```python
def validate_temporal_model(model_class, param_grid, train_df, test_df):
    """Complete validation pipeline with parameter search"""
    
    best_score = -1
    best_params = None
    best_results = None
    
    # Parameter search
    from itertools import product
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in product(*param_grid.values())]
    
    for params in param_combinations:
        print(f"Testing parameters: {params}")
        
        rolling_model = RollingWindowModel(
            model_class=model_class,
            model_params=params,
            test_window_months=12,
            retrain_window_months=12
        )
        
        results = rolling_model.fit_predict_rolling(
            train_df=train_df,
            test_df=test_df,
            verbose=False
        )
        
        f1_score = results['overall_metrics']['overall_f1']
        
        if f1_score > best_score:
            best_score = f1_score
            best_params = params
            best_results = results
    
    return best_params, best_results

# Example parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# Run validation
best_params, best_results = validate_temporal_model(
    RandomForestClassifier, param_grid, train_df, test_df
)

print(f"Best parameters: {best_params}")
print(f"Best F1 score: {best_results['overall_metrics']['overall_f1']:.4f}")
```

## Parameters

### RollingWindowModel Constructor
- `model_class`: sklearn-compatible model class (e.g., RandomForestClassifier)
- `model_params`: Dictionary of parameters for model initialization
- `test_window_months`: Number of months in each test window (int, default: 12)
- `retrain_window_months`: Number of months to add to training data after each test (int, default: 12)

### fit_predict_rolling() Method
- `train_df`: Initial training DataFrame with Country, Date, and feature columns
- `test_df`: Test DataFrame for rolling evaluation
- `train_labels`: Optional pre-computed labels for training data (np.array)
- `test_labels`: Optional pre-computed labels for test data (np.array)
- `verbose`: Whether to print progress information (bool, default: True)

## Key Methods

### RollingWindowModel Methods
- `fit_predict_rolling()`: Main method for rolling window evaluation
- `get_feature_importances(model_idx, top_n)`: Get feature importance from specific model
- `predict(X_test)`: Make predictions using the latest trained model
- `get_predictions_for_plotting()`: Get all predictions in plottable format
- `print_summary()`: Print comprehensive evaluation summary

### RollingWindowModelAdapter Methods
- `predict(X_test)`: sklearn-style prediction method
- `predict_proba(X_test)`: sklearn-style probability prediction method

## Workflow Details

### Rolling Window Process

1. **Initial Training**: Train model on historical training data
2. **First Test Window**: Test on first N months of test data
3. **Retrain**: Add first N months of test data to training set
4. **Next Test Window**: Test on next N months
5. **Repeat**: Continue until all test data is processed

### Data Flow
```
Historical Data → Model 1 → Test Window 1 → Retrain
                     ↓
Historical + Window 1 → Model 2 → Test Window 2 → Retrain
                            ↓
Historical + Windows 1-2 → Model 3 → Test Window 3 → ...
```

### Label Generation
If labels are not provided, the system automatically generates crisis labels using:
- `build_labels()` function from data utilities
- Configurable crisis definition parameters
- Consistent labeling across train and test periods

## Features

### Temporal Integrity
- **No Look-ahead Bias**: Strict temporal ordering prevents future information leakage
- **Realistic Evaluation**: Mimics real-world deployment scenarios
- **Progressive Training**: Models improve with more recent data

### Comprehensive Evaluation
- **Window-level Metrics**: Performance tracking for each test window
- **Overall Metrics**: Aggregated performance across all windows
- **Feature Evolution**: Track feature importance changes over time
- **Crisis Rate Analysis**: Monitor predicted vs. actual crisis rates

### Robust Data Handling
- **Missing Data**: Automatic handling of gaps in time series
- **Feature Engineering**: Consistent feature extraction across windows
- **Index Tracking**: Maintain alignment between predictions and ground truth

## Advanced Features

### Flexible Window Sizing
- Configure test window size based on prediction horizon needs
- Adjust retrain frequency for computational efficiency
- Support for overlapping or non-overlapping windows

### Model Persistence
- Store trained models from each window for analysis
- Feature importance tracking across time
- Model performance evolution monitoring

### Integration Capabilities
- Compatible with any sklearn-style model
- Support for custom model classes
- Integration with hyperparameter optimization

## Best Practices

### Window Size Selection
- **Short Windows (3-6 months)**: For rapid model adaptation
- **Medium Windows (6-12 months)**: Balance between stability and adaptation
- **Long Windows (12+ months)**: For stable, long-term predictions

### Data Requirements
- Minimum 2-3 years of historical training data
- At least 2-3 years of test data for robust evaluation
- Consistent feature availability across time periods

### Model Selection
- Tree-based models (RF, XGBoost) work well with tabular financial data
- Neural networks may require longer training windows
- Ensemble methods provide robust performance across windows

### Performance Monitoring
- Monitor both aggregate and window-specific metrics
- Track feature importance stability over time
- Watch for degrading performance in later windows

## Dependencies

- pandas: Time series data manipulation
- numpy: Numerical computations
- scikit-learn: Model evaluation metrics
- dateutil: Date manipulation utilities
- warnings: Warning control
- Data utilities: Label generation functions

## Common Use Cases

### Financial Crisis Prediction
- Quarterly model updates with annual retraining
- Monthly predictions with quarterly model updates
- Real-time monitoring with rolling validation

### Risk Management
- Regular model performance assessment
- Feature stability monitoring
- Stress testing across different time periods

### Model Development
- Temporal cross-validation for model selection
- Hyperparameter optimization with time-aware validation
- Performance benchmarking against baseline models

## Troubleshooting

### Common Issues
- **Insufficient Data**: Ensure adequate training periods for each window
- **Date Format Problems**: Verify consistent date formatting across datasets
- **Memory Issues**: Consider reducing model complexity for large datasets
- **Performance Degradation**: Monitor for concept drift and retrain more frequently

### Performance Tips
- Use efficient models for large numbers of windows
- Consider parallel processing for independent windows
- Monitor memory usage during long rolling evaluations
- Cache intermediate results for faster re-runs
