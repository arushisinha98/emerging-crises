import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed

def set_device(device="cuda", idx=0):
    """Set device for training"""
    if device != "cpu":
        if torch.cuda.device_count() > idx and torch.cuda.is_available():
            print(f"CUDA installed! Running on GPU {idx} {torch.cuda.get_device_name(idx)}!")
            device = f"cuda:{idx}"
        elif torch.cuda.device_count() > 0 and torch.cuda.is_available():
            print(f"CUDA installed but only {torch.cuda.device_count()} GPU(s) available! Running on GPU 0 {torch.cuda.get_device_name()}!")
            device = "cuda:0"
        else:
            device = "cpu"
            print("No GPU available! Running on CPU")
    return device

def worker_init_fn(worker_id):
    """Initialize worker with different seed"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def add_gaussian_noise(x, noise_factor=0.01):
    return x + torch.randn_like(x) * noise_factor

def get_save_path(params):
    hidden_dims_str = "_".join(map(str, params['hidden_dims']))
    noise_str = f"noise{params['noise_factor']}" if params['Gaussian_noise'] else "nonoise"
    model_name = f"FNN_bs{params['batch_size']}_hd{hidden_dims_str}_lr{params['learning_rate']}_{noise_str}"
    model_path = f"{model_name}.pth"
    training_data_path = f"{model_name}_training_data.pkl"
    return model_path, training_data_path

def train_epoch(model, train_loader, criterion, optimizer, device, use_noise=True, noise_factor=0.01):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for _, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        # Apply Gaussian noise during training
        if use_noise:
            noisy_data = add_gaussian_noise(data, noise_factor)
        else:
            noisy_data = data
        
        # Outputs from model: logits and features
        output, _ = model(noisy_data)
        output = output.squeeze()
        
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=0.5,
            norm_type=2
        )
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
    
    return total_loss / total_samples

def validate_epoch(model, val_loader, criterion, device, return_predictions=False):
    """Validate model for one epoch"""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Unpack batch as list of (features, target)
            features_input, target = batch
            features_input = features_input.to(device)
            target = target.to(device)
            
            # Outputs from model: logits and features
            output, _ = model(features_input)
            output = output.squeeze()
            
            loss = criterion(output, target)
            
            total_loss += loss.item() * features_input.size(0)
            total_samples += features_input.size(0)
            
            # Convert logits to probabilities
            probabilities = torch.sigmoid(output)
            
            # Store predictions and targets
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics (unchanged)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    pred_binary = (all_probabilities > 0.5).astype(int)
    
    metrics = {
        'loss': total_loss / total_samples,
        'accuracy': accuracy_score(all_targets, pred_binary),
        'precision': precision_score(all_targets, pred_binary, zero_division=0),
        'recall': recall_score(all_targets, pred_binary, zero_division=0),
        'f1': f1_score(all_targets, pred_binary, zero_division=0),
        'auc': roc_auc_score(all_targets, all_probabilities)
    }
    
    if return_predictions:
        return metrics, all_targets, pred_binary, all_probabilities
    else:
        return metrics

def plot_training_history(train_losses, val_losses, val_metrics_history, 
                          fold_num=None, save_path=None, show_plot=True):
    """
    Plot training and validation losses along with other metrics over epochs
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch  
        val_metrics_history: Dict with lists of validation metrics per epoch
        fold_num: Fold number for title (optional)
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    
    epochs = range(1, len(train_losses) + 1)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training vs Validation Loss
    axes[0, 0].plot(epochs, train_losses, 'bo-', label='Training Loss', linewidth=2, markersize=4)
    axes[0, 0].plot(epochs, val_losses, 'ro-', label='Validation Loss', linewidth=2, markersize=4)
    axes[0, 0].set_title('Training vs Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy and F1-Score
    if 'accuracy' in val_metrics_history and 'f1' in val_metrics_history:
        axes[0, 1].plot(epochs, val_metrics_history['accuracy'], 'go-', 
                       label='Accuracy', linewidth=2, markersize=4)
        axes[0, 1].plot(epochs, val_metrics_history['f1'], 'mo-', 
                       label='F1-Score', linewidth=2, markersize=4)
        axes[0, 1].set_title('Validation Accuracy & F1-Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
    
    # Plot 3: Precision and Recall
    if 'precision' in val_metrics_history and 'recall' in val_metrics_history:
        axes[1, 0].plot(epochs, val_metrics_history['precision'], 'co-', 
                       label='Precision', linewidth=2, markersize=4)
        axes[1, 0].plot(epochs, val_metrics_history['recall'], 'yo-', 
                       label='Recall', linewidth=2, markersize=4)
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)

    # Plot 4: AUC-ROC
    if 'auc' in val_metrics_history:
        axes[1, 1].plot(epochs, val_metrics_history['auc'], 'ko-', 
                       label='AUC-ROC', linewidth=2, markersize=4)
        axes[1, 1].set_title('AUC-ROC Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
    
    # Add main title
    if fold_num is not None:
        fig.suptitle(f'Training History - Fold {fold_num}', fontsize=16, fontweight='bold')
    else:
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def extract_embedding(model, data_loader, device):
    """Extract features from the penultimate layer for visualization"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Outputs from model: logits and features
            _, features = model(inputs)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    embeddings = np.concatenate(all_features, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    return embeddings, labels_array