import os
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

import wandb


class GMFTrainer:
    """Trainer handling the joint optimization of GMF generator, estimator, and decoder."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: Optimizer,
        scheduler: Optional[ReduceLROnPlateau],
        data_handler,
        config: Dict[str, Any],
        save_dir: str,
        wandb_run=None,
    ) -> None:
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_handler = data_handler
        self.config = config
        self.save_dir = save_dir
        self.run = wandb_run

        self.criterion = nn.MSELoss()
        self.gmf_weight = float(config.get('gmf_loss_weight', 1.0))
        self.decoder_weight = float(config.get('decoder_loss_weight', 1.0))

        self.label_mean_tensor = torch.tensor(self.data_handler.label_mean, device=self.device)
        self.label_std_tensor = torch.tensor(self.data_handler.label_std, device=self.device)

        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.best_checkpoint_path: Optional[str] = None

        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, 'model_architecture.txt'), 'w') as arch_file:
            arch_file.write(str(self.model))

    def _compute_accuracy(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        preds_denorm = preds * self.label_std_tensor + self.label_mean_tensor
        targets_denorm = targets * self.label_std_tensor + self.label_mean_tensor
        absolute_error = torch.abs(preds_denorm - targets_denorm)
        threshold = 0.05
        accurate_predictions = (absolute_error <= threshold).float()
        return accurate_predictions.mean().item() * 100.0

    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        train: bool = True,
    ) -> Dict[str, float]:
        inputs, targets, params = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        params = params.to(self.device)

        if train:
            self.optimizer.zero_grad()

        gmf_generated = self.model.generate_gmf(params, targets)
        gmf_estimated = self.model.estimator(inputs)
        decoded_from_generator = self.model.decode(params, gmf_generated)
        decoded_from_estimator = self.model.decode(params, gmf_estimated)

        loss_gmf = self.criterion(gmf_estimated, gmf_generated)
        loss_decoder = self.criterion(decoded_from_generator, targets)
        loss = self.gmf_weight * loss_gmf + self.decoder_weight * loss_decoder

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        preds_denorm = decoded_from_estimator * self.label_std_tensor + self.label_mean_tensor
        targets_denorm = targets * self.label_std_tensor + self.label_mean_tensor
        mse = torch.mean((preds_denorm - targets_denorm) ** 2).item()
        rmse = float(np.sqrt(mse))
        accuracy = self._compute_accuracy(decoded_from_estimator, targets)

        return {
            'loss': loss.item(),
            'gmf_loss': loss_gmf.item(),
            'decoder_loss': loss_decoder.item(),
            'rmse': rmse,
            'accuracy': accuracy,
        }

    def train_epoch(self, train_loader) -> Dict[str, float]:
        self.model.train()
        metrics = {'loss': 0.0, 'gmf_loss': 0.0, 'decoder_loss': 0.0, 'rmse': 0.0, 'accuracy': 0.0}
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, desc='Train')

        for batch_idx, batch in enumerate(train_loader, start=1):
            batch_metrics = self._step(batch, train=True)
            for key in metrics:
                metrics[key] += batch_metrics[key]
            batch_bar.set_postfix({k: f"{metrics[k] / batch_idx:.04f}" for k in ['loss', 'rmse']})
            batch_bar.update()

        batch_bar.close()
        for key in metrics:
            metrics[key] /= len(train_loader)
        return metrics

    def eval_epoch(self, val_loader) -> Dict[str, float]:
        self.model.eval()
        metrics = {'loss': 0.0, 'gmf_loss': 0.0, 'decoder_loss': 0.0, 'rmse': 0.0, 'accuracy': 0.0}
        batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, desc='Val')

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader, start=1):
                batch_metrics = self._step(batch, train=False)
                for key in metrics:
                    metrics[key] += batch_metrics[key]
                batch_bar.set_postfix({k: f"{metrics[k] / batch_idx:.04f}" for k in ['loss', 'rmse']})
                batch_bar.update()

        batch_bar.close()
        for key in metrics:
            metrics[key] /= len(val_loader)
        return metrics

    def test(self, test_loader) -> Dict[str, float]:
        self.model.eval()
        metrics = {'rmse': 0.0, 'mae': 0.0}
        batches = 0

        with torch.no_grad():
            for inputs, targets, params in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                params = params.to(self.device)

                gmf_estimated = self.model.estimator(inputs)
                decoded = self.model.decode(params, gmf_estimated)

                preds_denorm = decoded * self.label_std_tensor + self.label_mean_tensor
                targets_denorm = targets * self.label_std_tensor + self.label_mean_tensor

                diff = preds_denorm - targets_denorm
                mse = torch.mean(diff ** 2).item()
                mae = torch.mean(torch.abs(diff)).item()

                metrics['rmse'] += np.sqrt(mse)
                metrics['mae'] += mae
                batches += 1

        if batches > 0:
            metrics = {k: v / batches for k, v in metrics.items()}
        return metrics

    def save_checkpoint(self, epoch: int) -> None:
        checkpoint_path = os.path.join(self.save_dir, f'gmf_model_epoch_{epoch}.pt')
        torch.save({'model_state_dict': self.model.state_dict(), 'epoch': epoch}, checkpoint_path)
        self.best_checkpoint_path = checkpoint_path
        if self.run is not None:
            artifact = wandb.Artifact('gmf_model', type='model')
            artifact.add_file(checkpoint_path)
            self.run.log_artifact(artifact)

    def load_best_model(self) -> Optional[int]:
        if self.best_checkpoint_path is None or not os.path.exists(self.best_checkpoint_path):
            return None
        checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('epoch')

    def fit(self, train_loader, val_loader) -> None:
        for epoch in range(self.config['epochs']):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.eval_epoch(val_loader)

            if self.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/rmse': train_metrics['rmse'],
                    'val/loss': val_metrics['loss'],
                    'val/rmse': val_metrics['rmse'],
                    'train/gmf_loss': train_metrics['gmf_loss'],
                    'val/gmf_loss': val_metrics['gmf_loss'],
                })

            if self.scheduler is not None:
                self.scheduler.step(val_metrics['loss'])

            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                self.save_checkpoint(epoch)
