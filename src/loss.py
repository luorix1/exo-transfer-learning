import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """Mean Squared Error loss with optional importance weighting."""
    
    def __init__(self, weight_importance=False):
        super().__init__()
        self.weight_importance = weight_importance

    def forward(self, y_pred, y_true, y_var=None):
        """
        Compute MSE loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            y_var: Optional variance for importance weighting
        """
        err = torch.abs(y_pred - y_true)
        
        if self.weight_importance and y_var is not None:
            err = err / y_var
            
        return torch.mean(err**2)


class SmoothL1Loss(nn.Module):
    """Smooth L1 (Huber) loss with optional importance weighting."""
    
    def __init__(self, weight_importance=False):
        super().__init__()
        self.weight_importance = weight_importance

    def forward(self, y_pred, y_true, y_var=None):
        """
        Compute Smooth L1 loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            y_var: Optional variance for importance weighting
        """
        err = torch.abs(y_pred - y_true)
        
        if self.weight_importance and y_var is not None:
            err = err / y_var

        # Huber loss
        l1 = err - 0.5
        l2 = 0.5 * (err**2)
        mask = err < 1

        huber = l1
        huber[mask] = l2[mask]
        
        return torch.mean(huber)


class MAELoss(nn.Module):
    """Mean Absolute Error loss."""
    
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """Compute MAE loss."""
        return torch.mean(torch.abs(y_pred - y_true))


class JointMomentLoss(nn.Module):
    """Loss function for joint moment prediction."""
    
    def __init__(self, loss_type='mse'):
        super().__init__()
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'mae':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, y_pred, y_true):
        """Compute loss for joint moment prediction."""
        return self.loss_fn(y_pred, y_true)