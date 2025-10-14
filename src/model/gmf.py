import torch
import torch.nn as nn
from typing import Tuple


class GMFGenerator(nn.Module):
    """Generates generalized moment features (GMF) from subject parameters and joint moments."""

    def __init__(
        self,
        moment_size: int,
        param_size: int,
        gmf_size: int,
        hidden_size: int = 32,
        hidden_layers: int = 5,
    ) -> None:
        super().__init__()
        layers = []
        input_dim = moment_size + param_size
        for layer_idx in range(hidden_layers):
            in_features = input_dim if layer_idx == 0 else hidden_size
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
        layers.append(nn.Linear(hidden_size, gmf_size))
        self.network = nn.Sequential(*layers)

    def forward(self, params: torch.Tensor, moments: torch.Tensor) -> torch.Tensor:
        if params.dim() == 1:
            params = params.unsqueeze(0)
        if moments.dim() == 1:
            moments = moments.unsqueeze(0)
        x = torch.cat([moments, params], dim=-1)
        return self.network(x)


class GMFEstimator(nn.Module):
    """Estimates GMF directly from IMU windows using a GRU-based encoder."""

    def __init__(self, input_size: int, gmf_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.input_size = input_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, gmf_size)

    def forward(self, imu_window: torch.Tensor) -> torch.Tensor:
        if imu_window.dim() != 3:
            raise ValueError(
                f"Expected IMU window tensor of shape (batch, channels, time), got {imu_window.shape}"
            )
        # Convert to (batch, time, channels) for GRU
        x = imu_window.permute(0, 2, 1)
        _, hidden = self.gru(x)
        hidden_state = hidden[-1]
        return self.output_layer(hidden_state)


class GMFDecoder(nn.Module):
    """Decodes GMF back to joint moments while conditioning on subject parameters."""

    def __init__(
        self,
        gmf_size: int,
        param_size: int,
        output_size: int,
        hidden_size: int = 32,
        hidden_layers: int = 5,
    ) -> None:
        super().__init__()
        layers = []
        input_dim = gmf_size + param_size
        for layer_idx in range(hidden_layers):
            in_features = input_dim if layer_idx == 0 else hidden_size
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, params: torch.Tensor, gmf: torch.Tensor) -> torch.Tensor:
        if params.dim() == 1:
            params = params.unsqueeze(0)
        if gmf.dim() == 1:
            gmf = gmf.unsqueeze(0)
        x = torch.cat([gmf, params], dim=-1)
        return self.network(x)


class GMFModel(nn.Module):
    """Container module holding the GMF generator, estimator, and decoder."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        gmf_size: int,
        generator_hidden_size: int = 32,
        generator_hidden_layers: int = 5,
        estimator_hidden_size: int = 16,
        decoder_hidden_size: int = 32,
        decoder_hidden_layers: int = 5,
        param_size: int = 2,
    ) -> None:
        super().__init__()
        self.generator = GMFGenerator(
            moment_size=output_size,
            param_size=param_size,
            gmf_size=gmf_size,
            hidden_size=generator_hidden_size,
            hidden_layers=generator_hidden_layers,
        )
        self.estimator = GMFEstimator(
            input_size=input_size,
            gmf_size=gmf_size,
            hidden_size=estimator_hidden_size,
        )
        self.decoder = GMFDecoder(
            gmf_size=gmf_size,
            param_size=param_size,
            output_size=output_size,
            hidden_size=decoder_hidden_size,
            hidden_layers=decoder_hidden_layers,
        )

    def forward(self, imu_window: torch.Tensor, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gmf_est = self.estimator(imu_window)
        decoded_moment = self.decoder(params, gmf_est)
        return gmf_est, decoded_moment

    def generate_gmf(self, params: torch.Tensor, moments: torch.Tensor) -> torch.Tensor:
        return self.generator(params, moments)

    def decode(self, params: torch.Tensor, gmf: torch.Tensor) -> torch.Tensor:
        return self.decoder(params, gmf)
