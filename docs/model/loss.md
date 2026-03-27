# Custom Loss Functions Module

## Overview

The `loss.py` module provides specialized loss functions for handling class imbalance and improving model performance in financial crisis prediction. It includes various implementations of Focal Loss and other advanced loss functions designed for binary classification tasks with imbalanced datasets.

## Purpose

- Address class imbalance in financial crisis datasets
- Provide loss functions that focus on hard-to-classify examples
- Implement adaptive weighting schemes for better minority class detection
- Support label smoothing and regularization techniques
- Offer precision-focused loss functions for reducing false positives

## Main Classes

### FocalLoss

Standard Focal Loss implementation for addressing class imbalance by down-weighting easy examples.

#### Key Features
- Configurable alpha and gamma parameters
- Automatic probability handling for logits or probabilities
- Multiple reduction options (mean, sum, none)
- Robust handling of edge cases

### AdaptiveFocalLoss

Enhanced focal loss with adaptive parameters and label smoothing.

#### Key Features
- Adaptive gamma based on prediction confidence
- Built-in label smoothing for regularization
- Dynamic weighting for very hard examples
- Improved convergence properties

### WeightedBCELoss

Weighted Binary Cross-Entropy Loss for simple class balancing.

#### Key Features
- Positive class weighting
- Direct integration with PyTorch BCE with logits
- Simple and efficient implementation
- Good baseline for imbalanced datasets

### PrecisionFocalLoss

Focal loss variant with additional false positive penalty for high-precision requirements.

#### Key Features
- Additional penalty for false positives
- Configurable precision weighting
- Maintains focal loss benefits
- Optimized for precision-critical applications

## Usage Examples

### Basic Focal Loss

```python
from src.model.loss import FocalLoss
import torch
import torch.nn as nn

# Initialize focal loss
focal_loss = FocalLoss(
    alpha=0.25,      # Weight for positive class
    gamma=2.0,       # Focusing parameter
    reduction='mean'  # Reduction method
)

# Example training loop
model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch_idx, (data, targets) in enumerate(train_loader):
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(data)  # Raw logits (no sigmoid)
    
    # Calculate focal loss
    loss = focal_loss(logits, targets.float())
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if batch_idx % 100 == 0:
        print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
```

### Adaptive Focal Loss

```python
from src.model.loss import AdaptiveFocalLoss

# Enhanced focal loss with adaptive features
adaptive_focal = AdaptiveFocalLoss(
    alpha=0.9,              # Higher alpha for more imbalanced data
    gamma=2.5,              # Slightly higher gamma
    label_smoothing=0.1     # Label smoothing for regularization
)

# Training with adaptive loss
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for data, targets in train_loader:
        optimizer.zero_grad()
        
        outputs = model(data)
        loss = adaptive_focal(outputs, targets.float())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')
```

### Weighted BCE Loss

```python
from src.model.loss import WeightedBCELoss
import torch

# Calculate positive class weight based on class distribution
pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()])

# Initialize weighted BCE loss
weighted_bce = WeightedBCELoss(pos_weight=pos_weight)

# Simple training example
criterion = weighted_bce
for data, targets in train_loader:
    outputs = model(data)
    loss = criterion(outputs, targets.float())
    # ... rest of training loop
```

### Precision-Focused Loss

```python
from src.model.loss import PrecisionFocalLoss

# Loss function optimized for high precision
precision_loss = PrecisionFocalLoss(
    alpha=0.15,              # Lower alpha (less weight on positive class)
    gamma=2.5,               # Standard focusing parameter
    precision_weight=2.0     # Weight for false positive penalty
)

# Training for precision-critical application
for data, targets in train_loader:
    optimizer.zero_grad()
    
    predictions = model(data)
    loss = precision_loss(predictions, targets.float())
    
    loss.backward()
    optimizer.step()
```

### Comparing Loss Functions

```python
import matplotlib.pyplot as plt
import numpy as np

def compare_loss_functions(y_true, y_pred_range):
    """Compare different loss functions across prediction range"""
    
    losses = {
        'BCE': nn.BCEWithLogitsLoss(reduction='none'),
        'Focal (γ=2)': FocalLoss(alpha=0.25, gamma=2.0, reduction='none'),
        'Focal (γ=5)': FocalLoss(alpha=0.25, gamma=5.0, reduction='none'),
        'Adaptive Focal': AdaptiveFocalLoss(alpha=0.25, gamma=2.5),
        'Precision Focal': PrecisionFocalLoss(alpha=0.25, gamma=2.5)
    }
    
    y_true_tensor = torch.tensor([y_true], dtype=torch.float32)
    loss_values = {}
    
    for name, loss_fn in losses.items():
        values = []
        for pred in y_pred_range:
            pred_tensor = torch.tensor([pred], dtype=torch.float32)
            loss_val = loss_fn(pred_tensor, y_true_tensor)
            values.append(loss_val.item())
        loss_values[name] = values
    
    return loss_values

# Compare losses for positive class example
y_pred_range = np.linspace(-5, 5, 100)  # Logit range
loss_comparison = compare_loss_functions(y_true=1.0, y_pred_range=y_pred_range)

# Plot comparison
plt.figure(figsize=(12, 6))
for name, values in loss_comparison.items():
    plt.plot(y_pred_range, values, label=name, linewidth=2)

plt.xlabel('Predicted Logit')
plt.ylabel('Loss Value')
plt.title('Loss Function Comparison (Positive Class Example)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Dynamic Loss Selection

```python
def get_loss_function(dataset_balance, precision_critical=False):
    """Select appropriate loss function based on dataset characteristics"""
    
    # Calculate imbalance ratio
    positive_ratio = dataset_balance
    imbalance_ratio = positive_ratio / (1 - positive_ratio)
    
    if precision_critical:
        # Use precision-focused loss for critical applications
        return PrecisionFocalLoss(
            alpha=0.1,  # Lower weight on positive class
            gamma=2.0,
            precision_weight=3.0
        )
    elif imbalance_ratio < 0.1:  # Highly imbalanced
        # Use adaptive focal loss
        return AdaptiveFocalLoss(
            alpha=0.95,           # High alpha for extreme imbalance
            gamma=3.0,            # Higher gamma
            label_smoothing=0.05  # Light smoothing
        )
    elif imbalance_ratio < 0.3:  # Moderately imbalanced
        # Use standard focal loss
        return FocalLoss(
            alpha=0.75,
            gamma=2.0,
            reduction='mean'
        )
    else:  # Relatively balanced
        # Use weighted BCE
        pos_weight = torch.tensor([1.0 / positive_ratio])
        return WeightedBCELoss(pos_weight=pos_weight)

# Usage
positive_ratio = y_train.mean()  # Calculate from your data
loss_function = get_loss_function(positive_ratio, precision_critical=True)
```

### Integration with Model Training

```python
from src.model.architectures import FFNNClassifier

# Custom FFNN with focal loss
params = {
    'hidden_dims': [256, 128, 64],
    'dropout_rate': 0.3,
    'learning_rate': 1e-3,
    'batch_size': 128,
    'n_epochs': 100,
    'criterion': FocalLoss(alpha=0.25, gamma=2.0)  # Custom loss
}

# Train model with focal loss
ffnn = FFNNClassifier(params=params, seed=42)
ffnn.fit(train_data, train_labels)
```

## Parameters

### FocalLoss
- `alpha`: Weight for positive class (float, 0-1 or -1 to ignore, default: 0.25)
- `gamma`: Focusing parameter (float, default: 2.0)
- `reduction`: Reduction method ('mean', 'sum', 'none', default: 'mean')

### AdaptiveFocalLoss  
- `alpha`: Weight for positive class (float, default: 0.9)
- `gamma`: Base focusing parameter (float, default: 2.5)
- `label_smoothing`: Label smoothing factor (float, 0-1, default: 0.1)

### WeightedBCELoss
- `pos_weight`: Weight for positive class (torch.Tensor, optional)

### PrecisionFocalLoss
- `alpha`: Weight for positive class (float, default: 0.15)
- `gamma`: Focusing parameter (float, default: 2.5)
- `precision_weight`: Weight for false positive penalty (float, default: 2.0)

## Mathematical Background

### Focal Loss Formula
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

Where:
- `p_t`: Probability of the true class
- `α_t`: Class weighting factor
- `γ`: Focusing parameter (γ > 0)

### Key Properties
- **γ = 0**: Equivalent to standard cross-entropy
- **γ > 0**: Down-weights easy examples, focuses on hard examples
- **α**: Balances positive/negative class importance
- **Higher γ**: More focus on hard examples

### Adaptive Focal Loss
```
FL_adaptive(p_t) = -α_t * (1 - p_t)^(γ + 0.5*(1-p_t)) * log(p_t)
```
- Dynamically increases focusing for very hard examples

### Precision Focal Loss
```
PFL(p_t) = FL(p_t) + λ * p * (1 - y)
```
- Additional penalty for false positives
- `λ`: Precision weight parameter

## Advanced Usage Patterns

### Loss Scheduling

```python
class FocalLossScheduler:
    """Scheduler for dynamic focal loss parameters"""
    
    def __init__(self, initial_gamma=2.0, final_gamma=5.0, total_epochs=100):
        self.initial_gamma = initial_gamma
        self.final_gamma = final_gamma
        self.total_epochs = total_epochs
        self.focal_loss = FocalLoss(alpha=0.25, gamma=initial_gamma)
    
    def step(self, epoch):
        """Update gamma based on epoch"""
        progress = epoch / self.total_epochs
        current_gamma = self.initial_gamma + progress * (self.final_gamma - self.initial_gamma)
        self.focal_loss.gamma = current_gamma
        return self.focal_loss

# Usage in training loop
loss_scheduler = FocalLossScheduler(initial_gamma=1.0, final_gamma=3.0, total_epochs=200)

for epoch in range(200):
    current_loss_fn = loss_scheduler.step(epoch)
    
    for data, targets in train_loader:
        loss = current_loss_fn(model(data), targets.float())
        # ... rest of training
```

### Multi-Task Loss Combination

```python
def combined_loss(predictions, targets, recon_loss_weight=0.1):
    """Combine classification and reconstruction losses"""
    
    # Classification loss (focal loss)
    clf_loss = FocalLoss(alpha=0.25, gamma=2.0)(predictions, targets)
    
    # Reconstruction loss (if applicable for autoencoders)
    # recon_loss = F.mse_loss(reconstructed, original)
    
    # Combined loss
    # total_loss = clf_loss + recon_loss_weight * recon_loss
    
    return clf_loss  # Simplified for this example
```

## Best Practices

### Parameter Selection
- **Alpha**: Set based on class distribution (0.25 for 1:3 ratio, 0.1 for 1:9 ratio)
- **Gamma**: Start with 2.0, increase for more imbalance (up to 5.0)
- **Label Smoothing**: Use 0.05-0.1 for regularization
- **Precision Weight**: 1.5-3.0 for precision-critical tasks

### Training Tips
- Monitor both loss and metrics (precision, recall, F1)
- Use learning rate scheduling with focal loss
- Consider warmup period with lower gamma
- Validate on balanced validation set

### Common Issues
- **Vanishing Gradients**: Reduce gamma if training stalls
- **Overfitting**: Add label smoothing or reduce model complexity
- **Poor Calibration**: Consider temperature scaling post-training
- **Numerical Instability**: Ensure proper logit ranges

## Dependencies

- torch: PyTorch framework for tensor operations
- torch.nn: Neural network modules and loss functions
- torch.nn.functional: Functional implementations of loss functions

## Integration

Compatible with:
- All PyTorch models and training loops
- Custom architectures from architectures.py
- Automated hyperparameter tuning
- Multi-GPU training setups
- Mixed precision training

## Performance Considerations

### Computational Overhead
- Focal loss: ~20% slower than standard BCE
- Adaptive focal loss: ~30% slower than standard BCE
- Precision focal loss: ~25% slower than standard BCE
- Overhead usually negligible compared to model forward pass

### Memory Usage
- Minimal additional memory overhead
- Similar memory requirements to standard loss functions
- No significant impact on batch size limitations

## Troubleshooting

### Common Issues
- **NaN Losses**: Check for extreme logit values, consider gradient clipping
- **No Convergence**: Try lower gamma values or different alpha
- **Poor Performance**: Ensure proper class weighting and validation strategy
- **Unstable Training**: Add gradient clipping and learning rate scheduling
