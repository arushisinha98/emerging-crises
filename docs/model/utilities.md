# Model Training Utilities

## Overview

The `utilities.py` module provides essential utility functions for deep learning model training, reproducibility, device management, and training loop implementations. It includes functions for seed setting, device configuration, data augmentation, and comprehensive training/validation procedures.

## Purpose

- Ensure reproducible deep learning experiments
- Provide GPU/CPU device management utilities
- Implement robust training and validation procedures
- Support data augmentation and regularization techniques
- Offer comprehensive training history visualization

## Key Functions

### Reproducibility Functions
- `set_seed()`: Set all random seeds for reproducibility
- `worker_init_fn()`: Initialize data loader workers with different seeds

### Device Management Functions
- `set_device()`: Automatic GPU/CPU device selection and configuration

### Training Functions
- `train_epoch()`: Complete training epoch implementation
- `validate_epoch()`: Validation epoch with metrics calculation
- `plot_training_history()`: Training progress visualization

### Data Augmentation Functions
- `add_gaussian_noise()`: Apply Gaussian noise for regularization

### Utility Functions
- `get_save_path()`: Generate model save paths based on parameters
- `extract_embedding()`: Extract features from neural networks

## Usage Examples

### Basic Setup and Reproducibility

```python
from src.model.utilities import set_seed, set_device
import torch

# Set up reproducible environment
seed = set_seed(42)
device = set_device("cuda", idx=0)

print(f"Using seed: {seed}")
print(f"Using device: {device}")

# This ensures:
# - All random number generators use the same seed
# - PyTorch operations are deterministic
# - CUDA operations are reproducible
```

### Training Loop Implementation

```python
from src.model.utilities import train_epoch, validate_epoch, plot_training_history
import torch.nn as nn
import torch.optim as optim

# Initialize model, criterion, and optimizer
model = YourModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training configuration
num_epochs = 100
use_gaussian_noise = True
noise_factor = 0.01

# Training history tracking
train_losses = []
val_losses = []
val_metrics_history = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'auc': []
}

# Training loop
for epoch in range(num_epochs):
    # Training phase
    train_loss = train_epoch(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        use_noise=use_gaussian_noise,
        noise_factor=noise_factor
    )
    
    # Validation phase
    val_metrics = validate_epoch(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device
    )
    
    # Store history
    train_losses.append(train_loss)
    val_losses.append(val_metrics['loss'])
    
    for metric in val_metrics_history:
        val_metrics_history[metric].append(val_metrics[metric])
    
    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val F1: {val_metrics['f1']:.4f}")

# Visualize training history
plot_training_history(
    train_losses=train_losses,
    val_losses=val_losses,
    val_metrics_history=val_metrics_history,
    save_path="training_history.png",
    show_plot=True
)
```

### Advanced Training with Data Augmentation

```python
from src.model.utilities import add_gaussian_noise

# Custom training loop with noise augmentation
def advanced_train_epoch(model, train_loader, criterion, optimizer, device, 
                        noise_schedule=None, epoch=0):
    """Advanced training with scheduled noise"""
    
    model.train()
    total_loss = 0
    total_samples = 0
    
    # Dynamic noise scheduling
    if noise_schedule is not None:
        noise_factor = noise_schedule(epoch)
    else:
        noise_factor = 0.01
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Apply different augmentations randomly
        if torch.rand(1) > 0.5:  # 50% chance
            data = add_gaussian_noise(data, noise_factor)
        
        # Standard training step
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
    
    return total_loss / total_samples

# Noise scheduling function
def linear_noise_schedule(epoch, initial_noise=0.05, final_noise=0.01, total_epochs=100):
    """Linear decay of noise factor"""
    decay = (initial_noise - final_noise) / total_epochs
    return max(final_noise, initial_noise - epoch * decay)

# Use in training loop
for epoch in range(num_epochs):
    train_loss = advanced_train_epoch(
        model, train_loader, criterion, optimizer, device,
        noise_schedule=linear_noise_schedule, epoch=epoch
    )
```

### Cross-Validation Training

```python
from sklearn.model_selection import KFold
from src.model.utilities import plot_training_history

def cross_validation_training(model_class, train_dataset, model_params, 
                            training_params, k_folds=5, seed=42):
    """K-fold cross-validation training"""
    
    set_seed(seed)
    device = set_device("cuda")
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f"\nTraining Fold {fold + 1}/{k_folds}")
        
        # Create fold-specific data loaders
        from torch.utils.data import Subset, DataLoader
        
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=training_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=training_params['batch_size'], shuffle=False)
        
        # Initialize fresh model for this fold
        model = model_class(**model_params).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])
        
        # Train this fold
        fold_train_losses = []
        fold_val_losses = []
        fold_val_metrics = {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
        
        for epoch in range(training_params['n_epochs']):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = validate_epoch(model, val_loader, criterion, device)
            
            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_metrics['loss'])
            
            for metric in fold_val_metrics:
                fold_val_metrics[metric].append(val_metrics[metric])
        
        # Store fold results
        fold_results.append({
            'train_losses': fold_train_losses,
            'val_losses': fold_val_losses,
            'val_metrics': fold_val_metrics,
            'final_metrics': {metric: fold_val_metrics[metric][-1] for metric in fold_val_metrics}
        })
        
        # Plot this fold's training history
        plot_training_history(
            fold_train_losses, fold_val_losses, fold_val_metrics,
            fold_num=fold + 1,
            save_path=f"fold_{fold + 1}_training.png",
            show_plot=False
        )
    
    # Calculate average metrics across folds
    avg_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [fold['final_metrics'][metric] for fold in fold_results]
        avg_metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    return fold_results, avg_metrics

# Usage
cv_results, avg_performance = cross_validation_training(
    model_class=YourModelClass,
    train_dataset=dataset,
    model_params={'input_dim': 50, 'hidden_dim': 128},
    training_params={'batch_size': 32, 'learning_rate': 1e-3, 'n_epochs': 50},
    k_folds=5
)

print("Cross-Validation Results:")
for metric, stats in avg_performance.items():
    print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
```

### Feature Extraction and Analysis

```python
from src.model.utilities import extract_embedding

# Extract embeddings from trained model
embeddings, labels = extract_embedding(
    model=trained_model,
    data_loader=test_loader,
    device=device
)

print(f"Extracted embeddings shape: {embeddings.shape}")
print(f"Labels shape: {labels.shape}")

# Visualize embeddings
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot embeddings colored by labels
plt.figure(figsize=(10, 8))
colors = ['blue', 'red']
for label in np.unique(labels):
    mask = labels == label
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                c=colors[int(label)], label=f'Class {int(label)}', alpha=0.6)

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('Neural Network Embeddings Visualization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Model Saving and Loading with Utilities

```python
from src.model.utilities import get_save_path

# Generate save paths based on model parameters
params = {
    'batch_size': 32,
    'hidden_dims': [256, 128, 64],
    'learning_rate': 1e-3,
    'Gaussian_noise': True,
    'noise_factor': 0.01
}

model_path, training_data_path = get_save_path(params)
print(f"Model will be saved to: {model_path}")
print(f"Training data will be saved to: {training_data_path}")

# Save model and training history
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'val_metrics_history': val_metrics_history,
    'parameters': params
}, model_path)

# Save additional training data
import pickle
with open(training_data_path, 'wb') as f:
    pickle.dump({
        'train_losses': train_losses,
        'val_metrics_history': val_metrics_history,
        'final_performance': val_metrics_history
    }, f)

# Load model and training data
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

with open(training_data_path, 'rb') as f:
    training_data = pickle.load(f)
```

### Multi-GPU Training Setup

```python
# Advanced device setup for multi-GPU training
def setup_multi_gpu_training(model, device_ids=None):
    """Setup model for multi-GPU training"""
    
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    
    if len(device_ids) > 1:
        print(f"Using {len(device_ids)} GPUs: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)
        main_device = f"cuda:{device_ids[0]}"
    elif torch.cuda.is_available():
        main_device = "cuda:0"
        print("Using single GPU")
    else:
        main_device = "cpu"
        print("Using CPU")
    
    model = model.to(main_device)
    return model, main_device

# Usage
model, device = setup_multi_gpu_training(model, device_ids=[0, 1, 2])

# DataLoader with multi-worker support
from src.model.utilities import worker_init_fn

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    worker_init_fn=worker_init_fn,  # Ensures different seeds per worker
    pin_memory=True if torch.cuda.is_available() else False
)
```

## Parameters

### set_seed()
- `seed`: Random seed value (int, default: 42)
- Returns: The seed value used

### set_device()
- `device`: Preferred device type ("cuda" or "cpu", default: "cuda")
- `idx`: GPU index to use (int, default: 0)
- Returns: Device string (e.g., "cuda:0", "cpu")

### train_epoch()
- `model`: PyTorch model to train
- `train_loader`: Training data loader
- `criterion`: Loss function
- `optimizer`: Optimizer instance
- `device`: Device to run on
- `use_noise`: Whether to apply Gaussian noise (bool, default: True)
- `noise_factor`: Standard deviation of noise (float, default: 0.01)
- Returns: Average training loss

### validate_epoch()
- `model`: PyTorch model to validate
- `val_loader`: Validation data loader
- `criterion`: Loss function
- `device`: Device to run on
- `return_predictions`: Whether to return predictions (bool, default: False)
- Returns: Dictionary of metrics or tuple with predictions

### plot_training_history()
- `train_losses`: List of training losses per epoch
- `val_losses`: List of validation losses per epoch
- `val_metrics_history`: Dictionary of validation metrics per epoch
- `fold_num`: Fold number for cross-validation (int, optional)
- `save_path`: Path to save plot (str, optional)
- `show_plot`: Whether to display plot (bool, default: True)

## Key Features

### Reproducibility
- **Complete Seed Setting**: Sets seeds for Python, NumPy, PyTorch, and CUDA
- **Deterministic Operations**: Ensures reproducible results across runs
- **Worker Initialization**: Proper seed handling for multi-worker data loading
- **CUDA Determinism**: Enables deterministic CUDA operations

### Device Management
- **Automatic Detection**: Automatically detects and configures available GPUs
- **Graceful Fallback**: Falls back to CPU if GPU is not available
- **Multi-GPU Support**: Handles multiple GPU setups
- **Device Information**: Provides detailed device information

### Training Infrastructure
- **Robust Training Loops**: Production-ready training and validation procedures
- **Gradient Clipping**: Built-in gradient clipping for stability
- **Comprehensive Metrics**: Calculates accuracy, precision, recall, F1, and AUC
- **Progress Tracking**: Detailed progress reporting with tqdm

### Data Augmentation
- **Gaussian Noise**: Configurable noise injection for regularization
- **Flexible Application**: Apply augmentation selectively during training
- **Scheduled Augmentation**: Support for dynamic augmentation schedules

## Advanced Features

### Training Enhancements
- **Mixed Precision**: Support for automatic mixed precision training
- **Learning Rate Scheduling**: Integration with PyTorch schedulers
- **Early Stopping**: Configurable early stopping based on validation metrics
- **Checkpoint Management**: Automatic model checkpointing

### Monitoring and Visualization
- **Real-time Metrics**: Live metric calculation during training
- **Comprehensive Plotting**: Multi-subplot training visualization
- **Export Capabilities**: Save plots and metrics for later analysis
- **Cross-validation Support**: Specialized visualization for k-fold CV

### Memory and Performance
- **Efficient Computation**: Optimized training loops for speed
- **Memory Management**: Proper memory cleanup and optimization
- **Batch Processing**: Efficient batch-wise operations
- **GPU Utilization**: Optimized GPU memory usage

## Best Practices

### Reproducibility
- Always set seeds at the beginning of experiments
- Use consistent seeds across related experiments
- Document seed values in experiment logs
- Consider multiple seeds for robust evaluation

### Device Management
- Check GPU availability before starting training
- Monitor GPU memory usage during training
- Use appropriate batch sizes for available memory
- Consider mixed precision for large models

### Training Configuration
- Start with conservative learning rates
- Use gradient clipping for RNN-based models
- Monitor both training and validation metrics
- Save checkpoints regularly for long training runs

### Performance Optimization
- Use multiple workers for data loading when possible
- Pin memory for faster GPU transfers
- Consider data loading bottlenecks
- Profile code to identify performance issues

## Dependencies

- torch: PyTorch framework
- numpy: Numerical computations
- matplotlib: Plotting and visualization
- tqdm: Progress bars
- scikit-learn: Metrics calculation
- random: Python random number generation

## Integration

### Framework Compatibility
- **PyTorch Lightning**: Compatible with Lightning training loops
- **Hugging Face**: Works with Transformers training procedures
- **Custom Frameworks**: Easy integration with custom training systems

### Model Support
- **Architectures Module**: Direct integration with custom architectures
- **Standard Models**: Works with any PyTorch model
- **Multi-output Models**: Supports complex model outputs

### Pipeline Integration
- **Data Loading**: Seamless integration with dataset classes
- **Preprocessing**: Compatible with data preprocessing pipelines
- **Evaluation**: Works with evaluation and plotting utilities

## Common Patterns

### Experiment Management
```python
def run_experiment(config):
    # Setup
    set_seed(config['seed'])
    device = set_device(config['device'])
    
    # Model and training
    model = create_model(config).to(device)
    results = train_model(model, config, device)
    
    # Save results
    save_experiment_results(config, results)
    return results
```

### Hyperparameter Optimization
```python
def objective(trial):
    config = {
        'learning_rate': trial.suggest_loguniform('lr', 1e-5, 1e-1),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'noise_factor': trial.suggest_uniform('noise', 0.0, 0.1)
    }
    
    results = run_experiment(config)
    return results['best_val_score']
```

### Model Comparison
```python
def compare_models(model_configs, dataset):
    results = {}
    for name, config in model_configs.items():
        print(f"Training {name}...")
        results[name] = run_experiment(config)
    
    return results
```
