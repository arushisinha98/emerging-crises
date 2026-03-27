import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reference: https://docs.pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if not (0 <= alpha <= 1) and alpha != -1:
            raise ValueError(f"Invalid alpha value: {alpha}. alpha must be in the range [0,1] or -1 for ignore.")
    
    def forward(self, inputs, targets):
        """
        Calculate focal loss
        
        Args:
            inputs: predictions (logits or probabilities)
            targets: ground truth labels
        """
        # Flatten tensors if needed
        if inputs.dim() > 1:
            inputs = inputs.view(-1)
        if targets.dim() > 1:
            targets = targets.view(-1)
        
        # Convert logits to probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate cross entropy loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate p_t (probability of true class)
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Apply focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if specified
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class AdaptiveFocalLoss(nn.Module):
    """Enhanced focal loss with adaptive parameters"""
    
    def __init__(self, alpha=0.9, gamma=2.5, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        # Apply label smoothing for regularization
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Adaptive gamma based on prediction confidence
        adaptive_gamma = self.gamma + (1 - p_t) * 0.5  # Increase focus on very hard examples
        focal_weight = (1 - p_t) ** adaptive_gamma
        
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * ce_loss
        
        return focal_loss.mean()
    
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
    
class PrecisionFocalLoss(nn.Module):
    def __init__(self, alpha=0.15, gamma=2.5, precision_weight=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.precision_weight = precision_weight
        
    def forward(self, predictions, targets):
        # Standard focal loss
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        # False positive penalty for precision
        probs = torch.sigmoid(predictions)
        fp_penalty = self.precision_weight * (probs * (1 - targets))
        
        return (focal_loss + fp_penalty).mean()