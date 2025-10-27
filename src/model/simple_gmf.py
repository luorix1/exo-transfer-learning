import torch
import torch.nn as nn
from typing import Tuple


class SimpleGMFModel(nn.Module):
    """Simplified GMF model that directly predicts joint moments from IMU data."""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        param_size: int = 2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.param_size = param_size
        
        # IMU encoder (GRU-based)
        self.imu_encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Subject parameter encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(param_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined feature processor
        combined_size = hidden_size + hidden_size // 2
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(combined_size if i == 0 else hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(hidden_size, output_size))
        self.predictor = nn.Sequential(*layers)
        
    def forward(self, imu_window: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: IMU window + subject params -> joint moment prediction
        
        Args:
            imu_window: (batch, channels, time) - IMU data
            params: (batch, param_size) - subject parameters (mass, height)
        
        Returns:
            joint_moment: (batch, output_size) - predicted joint moment
        """
        if imu_window.dim() != 3:
            raise ValueError(f"Expected IMU window tensor of shape (batch, channels, time), got {imu_window.shape}")
        
        # Encode IMU data
        # Convert to (batch, time, channels) for GRU
        x = imu_window.permute(0, 2, 1)
        _, hidden = self.imu_encoder(x)
        imu_features = hidden[-1]  # Use last layer's hidden state
        
        # Encode subject parameters
        param_features = self.param_encoder(params)
        
        # Combine features
        combined = torch.cat([imu_features, param_features], dim=-1)
        
        # Predict joint moment
        joint_moment = self.predictor(combined)
        
        return joint_moment
    
    def predict_without_params(self, imu_window: torch.Tensor) -> torch.Tensor:
        """Predict without subject parameters (for testing)."""
        batch_size = imu_window.shape[0]
        dummy_params = torch.zeros(batch_size, self.param_size, device=imu_window.device)
        return self.forward(imu_window, dummy_params)
