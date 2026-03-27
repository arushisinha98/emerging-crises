# Hyperparameter Tuning Module

## Overview

The `tuning.py` module provides advanced hyperparameter optimization capabilities using Bayesian optimization. It's specifically designed for optimizing dimensionality reduction models combined with classification pipelines, enabling end-to-end hyperparameter tuning for complex machine learning workflows.

## Purpose

- Optimize hyperparameters for embedding + classification pipelines
- Provide Bayesian optimization for efficient parameter search
- Support various dimensionality reduction and classification model combinations
- Enable automated hyperparameter tuning with minimal manual intervention
- Handle complex parameter spaces with categorical and continuous variables

## Main Functions

### objective_function()

Core optimization function that trains and evaluates embedding + classification model combinations.

### run_bayesian_search()

Main interface for running Bayesian optimization with configurable search spaces and iterations.

### stratified_gridsearch()

Stratified grid search utility for classification model optimization within the pipeline.

## Usage Examples

### Basic Bayesian Optimization

```python
from src.features.vae import TimeSeriesVAE
from src.model.tuning import run_bayesian_search
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Load your data
train_data = pd.read_csv('financial_data.csv')
train_labels = np.array([0, 1, 0, 1, ...])  # Crisis labels

# Run Bayesian optimization for TimeSeriesVAE + RandomForest
results = run_bayesian_search(
    embed_model=TimeSeriesVAE,           # Embedding model class
    classify_model=RandomForestClassifier(), # Classification model
    train_data=train_data,
    train_labels=train_labels,
    n_iter=25,                          # Number of optimization iterations
    init_points=5,                      # Random initialization points
    seed=42                             # Reproducibility
)

print(f"Best parameters: {results['best_params']}")
print(f"Best score: {results['best_score']:.4f}")
```

### VAE + Classification Optimization

```python
from src.features.vae import BaseVAE
from xgboost import XGBClassifier

# Optimize BaseVAE with XGBoost classifier
vae_results = run_bayesian_search(
    embed_model=BaseVAE,
    classify_model=XGBClassifier(),
    train_data=train_data,
    train_labels=train_labels,
    n_iter=30,
    init_points=8,
    seed=42
)

# Extract optimized parameters
best_vae_params = vae_results['best_params']
print("Optimized VAE parameters:")
for param, value in best_vae_params.items():
    print(f"  {param}: {value}")

# Train final model with optimized parameters
from src.features.utilities import load_or_train_model

final_vae = load_or_train_model(
    model_class=BaseVAE,
    params=best_vae_params,
    train_data=train_data,
    train_labels=train_labels
)

# Transform data and train classifier
transformed_data = final_vae.transform(train_data)
final_classifier = XGBClassifier(**best_classifier_params)
final_classifier.fit(transformed_data, train_labels)
```

### U-Net VAE Optimization

```python
from src.features.unet import TimeSeriesUNET
from lightgbm import LGBMClassifier

# Optimize TimeSeriesUNET with LightGBM
unet_results = run_bayesian_search(
    embed_model=TimeSeriesUNET,
    classify_model=LGBMClassifier(),
    train_data=sequential_data,
    train_labels=train_labels,
    n_iter=20,
    init_points=5,
    seed=42
)

# The search will optimize parameters like:
# - hidden_size: LSTM hidden dimensions
# - n_components: Latent space dimensions
# - num_layers: Number of encoder/decoder layers
# - sequence_length: Input sequence length
# - Plus LGBMClassifier parameters
```

### Custom Search Space

```python
# For advanced users wanting to modify search spaces
# (The search spaces are defined within run_bayesian_search)

def custom_objective(embed_model, classify_model, train_data, train_labels, **kwargs):
    """Custom objective with additional constraints"""
    
    # Extract and validate parameters
    params = {
        'n_components': int(kwargs['n_components']),
        'learning_rate': kwargs['learning_rate'],
        'batch_size': int(kwargs['batch_size']),
        'beta': kwargs['beta']
    }
    
    # Add custom constraints
    if params['n_components'] > 20 and params['beta'] < 1.0:
        return -1e6  # Penalize this combination
    
    # Standard optimization logic
    from src.features.utilities import load_or_train_model
    from src.model.tuning import stratified_gridsearch
    
    try:
        # Train embedding model
        embedder = load_or_train_model(
            model_class=embed_model,
            params=params,
            train_data=train_data,
            train_labels=train_labels,
            force_retrain=True
        )
        
        # Optimize classifier
        param_grid = {'n_estimators': [50, 100, 150]}
        grid_search = stratified_gridsearch(
            param_grid, classify_model, 
            embedder.transform(train_data), train_labels
        )
        
        return -grid_search.best_score_  # Negative for maximization
        
    except Exception as e:
        print(f"Error in custom objective: {e}")
        return -1e6
```

### Analyzing Optimization Results

```python
# Detailed analysis of optimization results
import matplotlib.pyplot as plt

# Load results dataframe
results_df = vae_results['results_df']

# Plot optimization progress
plt.figure(figsize=(12, 8))

# Subplot 1: Optimization progress
plt.subplot(2, 2, 1)
plt.plot(results_df.index, results_df['target'], 'b-o')
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.title('Optimization Progress')
plt.grid(True, alpha=0.3)

# Subplot 2: Parameter correlation
plt.subplot(2, 2, 2)
if 'n_components' in results_df.columns and 'beta' in results_df.columns:
    plt.scatter(results_df['n_components'], results_df['beta'], 
                c=results_df['target'], cmap='viridis')
    plt.colorbar(label='Objective Value')
    plt.xlabel('n_components')
    plt.ylabel('beta')
    plt.title('Parameter Correlation')

# Subplot 3: Best parameters over time
plt.subplot(2, 2, 3)
best_so_far = results_df['target'].cummax()
plt.plot(results_df.index, best_so_far, 'r-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Best Score So Far')
plt.title('Best Score Evolution')
plt.grid(True, alpha=0.3)

# Subplot 4: Parameter distribution
plt.subplot(2, 2, 4)
if 'learning_rate' in results_df.columns:
    plt.hist(results_df['learning_rate'], bins=15, alpha=0.7)
    plt.xlabel('Learning Rate')
    plt.ylabel('Frequency')
    plt.title('Learning Rate Distribution')

plt.tight_layout()
plt.show()

# Print parameter statistics
print("\nParameter Statistics:")
numeric_cols = results_df.select_dtypes(include=[float, int]).columns
for col in numeric_cols:
    if col != 'target':
        best_idx = results_df['target'].idxmax()
        best_value = results_df.loc[best_idx, col]
        mean_value = results_df[col].mean()
        print(f"{col}: Best={best_value:.4f}, Mean={mean_value:.4f}")
```

### Multi-Stage Optimization

```python
def multi_stage_optimization(embed_models, classify_models, train_data, train_labels):
    """Optimize multiple model combinations"""
    
    results = {}
    
    for embed_name, embed_class in embed_models.items():
        for clf_name, clf_class in classify_models.items():
            
            combination_name = f"{embed_name}_{clf_name}"
            print(f"\nOptimizing {combination_name}...")
            
            # Optimize this combination
            combo_results = run_bayesian_search(
                embed_model=embed_class,
                classify_model=clf_class,
                train_data=train_data,
                train_labels=train_labels,
                n_iter=15,  # Fewer iterations per combination
                init_points=3,
                seed=42
            )
            
            results[combination_name] = {
                'best_score': combo_results['best_score'],
                'best_params': combo_results['best_params'],
                'embed_model': embed_class,
                'classify_model': clf_class
            }
    
    # Find overall best combination
    best_combo = max(results.keys(), key=lambda k: results[k]['best_score'])
    print(f"\nBest overall combination: {best_combo}")
    print(f"Best score: {results[best_combo]['best_score']:.4f}")
    
    return results, best_combo

# Define model combinations to test
embed_models = {
    'BaseVAE': BaseVAE,
    'TimeSeriesVAE': TimeSeriesVAE,
    'TimeSeriesUNET': TimeSeriesUNET
}

classify_models = {
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier()
}

# Run multi-stage optimization
all_results, best_combination = multi_stage_optimization(
    embed_models, classify_models, train_data, train_labels
)
```

### Optimization with Cross-Validation

```python
def cv_aware_optimization(embed_model, classify_model, train_data, train_labels, cv_folds=3):
    """Bayesian optimization with cross-validation"""
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    
    def cv_objective(**kwargs):
        # Extract parameters
        params = {
            'n_components': int(kwargs['n_components']),
            'learning_rate': kwargs['learning_rate'],
            'batch_size': int(kwargs['batch_size']),
            'beta': kwargs['beta'],
            'n_epochs': 25  # Fixed for speed
        }
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, train_labels)):
            try:
                # Split data
                fold_train_data = train_data.iloc[train_idx]
                fold_val_data = train_data.iloc[val_idx]
                fold_train_labels = train_labels[train_idx]
                fold_val_labels = train_labels[val_idx]
                
                # Train embedder
                embedder = load_or_train_model(
                    model_class=embed_model,
                    params=params,
                    train_data=fold_train_data,
                    train_labels=fold_train_labels,
                    force_retrain=True
                )
                
                # Transform data
                fold_train_transformed = embedder.transform(fold_train_data)
                fold_val_transformed = embedder.transform(fold_val_data)
                
                # Train classifier
                classifier = classify_model
                classifier.fit(fold_train_transformed, fold_train_labels)
                
                # Evaluate
                val_proba = classifier.predict_proba(fold_val_transformed)[:, 1]
                fold_score = roc_auc_score(fold_val_labels, val_proba)
                scores.append(fold_score)
                
            except Exception as e:
                print(f"Error in fold {fold}: {e}")
                scores.append(0.5)  # Random performance
        
        return -np.mean(scores)  # Negative for maximization
    
    # Run Bayesian optimization with CV objective
    # (Would need to implement custom optimization loop)
    print("CV-aware optimization would be implemented here")
    return {'cv_scores': scores, 'mean_score': np.mean(scores)}
```

## Parameters

### run_bayesian_search()
- `embed_model`: Embedding model class (VAE, TimeSeriesVAE, etc.)
- `classify_model`: Classification model instance
- `train_data`: Training DataFrame
- `train_labels`: Training labels array
- `n_iter`: Number of Bayesian optimization iterations (int, default: 25)
- `init_points`: Number of random initialization points (int, default: 5)
- `seed`: Random seed for reproducibility (int, default: 42)

### Search Spaces

The module defines different parameter bounds for different model types:

#### TimeSeriesVAE/TimeSeriesUNET
- `hidden_size`: (32, 256) - LSTM hidden units
- `n_components`: (2, 32) - Latent dimensions
- `num_layers`: (1, 3) - Number of layers
- `sequence_length`: (12, 48) - Input sequence length
- `batch_size`: (8, 32) - Training batch size
- `beta`: (1.0, 1.0) - VAE beta parameter (fixed)
- `learning_rate`: (1e-3, 1e-3) - Learning rate (fixed)
- `block_int`: (0, 1) - LSTM vs GRU selection

#### BaseVAE
- `hidden_dim1`: (32, 128) - First hidden layer size
- `hidden_dim2`: (16, 96) - Second hidden layer size
- `hidden_dim3`: (8, 64) - Third hidden layer size
- `n_components`: (2, 10) - Latent dimensions
- `batch_size`: (16, 128) - Training batch size
- `beta`: (0.5, 3.0) - VAE beta parameter
- `learning_rate`: (1e-5, 1e-3) - Learning rate

## Key Features

### Bayesian Optimization
- **Efficient Search**: Uses Gaussian processes to model objective function
- **Adaptive Sampling**: Focuses search on promising regions
- **Early Convergence**: Often finds good solutions with fewer evaluations
- **Handles Noise**: Robust to noisy objective functions

### End-to-End Pipeline
- **Integrated Optimization**: Optimizes both embedding and classification stages
- **Automatic Model Training**: Handles model instantiation and training
- **Pipeline Consistency**: Ensures compatible parameter combinations
- **Performance Tracking**: Comprehensive logging and result storage

### Flexible Architecture
- **Multiple Model Types**: Supports various embedding models (VAE, UNET, etc.)
- **Classifier Integration**: Works with any sklearn-compatible classifier
- **Custom Search Spaces**: Easily modify parameter bounds
- **Extensible Framework**: Add new model types with minimal changes

## Advanced Features

### Parameter Handling
- **Type Conversion**: Automatic conversion of continuous to discrete parameters
- **Categorical Variables**: Handles discrete choices (LSTM vs GRU)
- **Parameter Validation**: Ensures valid parameter combinations
- **Constraint Handling**: Built-in constraint satisfaction

### Result Management
- **Comprehensive Logging**: All trials saved to CSV
- **Result Analysis**: Built-in analysis and visualization support
- **Model Persistence**: Option to save best models
- **Reproducibility**: Full random seed control

### Error Handling
- **Graceful Failures**: Continues optimization despite individual failures
- **Exception Logging**: Detailed error reporting
- **Fallback Scores**: Assigns poor scores to failed trials
- **Robust Optimization**: Handles various failure modes

## Best Practices

### Optimization Strategy
- Start with fewer iterations (15-25) for initial exploration
- Use more initialization points (5-10) for complex search spaces
- Monitor convergence and extend iterations if needed
- Consider multiple random seeds for robustness

### Parameter Space Design
- Set realistic bounds based on computational constraints
- Include problem-specific constraints
- Balance exploration vs exploitation
- Consider parameter interactions

### Computational Efficiency
- Use fixed, reasonable values for expensive parameters (n_epochs)
- Consider parallel evaluation if resources allow
- Monitor memory usage during optimization
- Save intermediate results frequently

### Result Interpretation
- Analyze parameter correlations and interactions
- Validate best parameters on independent test set
- Consider ensemble of top configurations
- Document optimization settings and results

## Dependencies

- bayes_opt: Bayesian optimization library
- scikit-learn: Cross-validation and grid search
- numpy: Numerical computations
- pandas: Data manipulation and result storage
- Feature utilities: Model loading and management

## Integration

### Compatible Models
- **Embedding Models**: All VAE variants, PCA, UMAP, t-SNE
- **Classifiers**: All sklearn-compatible models
- **Custom Models**: Any model following the interface pattern

### Workflow Integration
- Feature engineering pipelines
- Cross-validation frameworks
- Model selection workflows
- Automated ML systems

## Performance Considerations

### Computational Cost
- Each iteration trains full embedding + classification pipeline
- Consider computational budget when setting n_iter
- GPU acceleration recommended for deep learning models
- Memory requirements scale with model complexity

### Optimization Efficiency
- Bayesian optimization typically 3-5x more efficient than grid search
- Good results often achieved with 20-50 evaluations
- Early stopping based on convergence detection
- Parallel evaluation can significantly reduce wall-clock time

## Common Issues and Solutions

### Convergence Problems
- **Issue**: Optimization doesn't improve after initial points
- **Solution**: Increase exploration by adding more init_points

### Memory Errors
- **Issue**: Out of memory during model training
- **Solution**: Reduce batch_size bounds or model complexity

### Poor Performance
- **Issue**: All trials perform poorly
- **Solution**: Check data quality and parameter bounds

### Inconsistent Results
- **Issue**: Different runs give very different results
- **Solution**: Increase n_iter and use multiple seeds
