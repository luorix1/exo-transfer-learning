import os
from typing import Dict, Any, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        optimizer_ge: Optimizer,
        optimizer_gd: Optimizer,
        scheduler_ge: Optional[ReduceLROnPlateau],
        scheduler_gd: Optional[ReduceLROnPlateau],
        data_handler,
        config: Dict[str, Any],
        save_dir: str,
        wandb_run=None,
    ) -> None:
        self.model = model
        self.device = device
        # Two optimizers: Phase A (Generator + Estimator), Phase B (Generator + Decoder)
        self.optimizer_ge = optimizer_ge
        self.optimizer_gd = optimizer_gd
        self.scheduler_ge = scheduler_ge
        self.scheduler_gd = scheduler_gd
        self.data_handler = data_handler
        self.config = config
        self.save_dir = save_dir
        self.run = wandb_run

        self.criterion = nn.MSELoss()
        # Loss weights: w1 for L1 (alignment), w2 for L2 (decodability)
        self.gmf_weight = float(config.get('gmf_loss_weight', 1.0))
        self.decoder_weight = float(config.get('decoder_loss_weight', 0.05))
        self.phaseA_decoder_coeff = float(config.get('phaseA_decoder_coeff', 0.0))
        self.warmup_epochs = int(config.get('warmup_epochs', 0))

        self.label_mean_tensor = torch.tensor(self.data_handler.label_mean, device=self.device)
        self.label_std_tensor = torch.tensor(self.data_handler.label_std, device=self.device)
        
        # Handle case where subject parameters might not be available
        if hasattr(self.data_handler, 'param_mean') and self.data_handler.param_mean is not None:
            self.param_mean_tensor = torch.tensor(self.data_handler.param_mean, device=self.device)
            self.param_std_tensor = torch.tensor(self.data_handler.param_std, device=self.device)
        else:
            self.param_mean_tensor = None
            self.param_std_tensor = None

        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.best_checkpoint_path: Optional[str] = None
        self.current_epoch = 0
        self.patience_counter = 0
        self.early_stopping_patience = 10

        self.train_accuracy_history = []
        self.val_accuracy_history = []
        self.train_rmse_history = []
        self.val_rmse_history = []

        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, 'model_architecture.txt'), 'w') as arch_file:
            arch_file.write(str(self.model))

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for param in module.parameters():
            param.requires_grad = requires_grad

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
        if len(batch) == 3:
            inputs, targets, params = batch
        else:
            inputs, targets = batch
            # Create dummy parameters if not available (mass=70kg, height=1.7m as defaults)
            params = torch.tensor([[70.0, 1.7]] * targets.shape[0], device=self.device)
        
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        params = params.to(self.device)
        
        # Normalize parameters if available
        if self.param_mean_tensor is not None and self.param_std_tensor is not None:
            params = (params - self.param_mean_tensor) / self.param_std_tensor

        # Forward pass
        gmf_estimated = self.model.estimator(inputs)
        decoded_from_estimator = self.model.decode(params, gmf_estimated)
        
        # Validate outputs
        if torch.isnan(decoded_from_estimator).any():
            print("WARNING: NaN detected in predictions!")
            decoded_from_estimator = torch.nan_to_num(decoded_from_estimator, nan=0.0)
        
        if torch.isnan(gmf_estimated).any():
            print("WARNING: NaN detected in GMF!")
            gmf_estimated = torch.nan_to_num(gmf_estimated, nan=0.0)
        
        # Compute main reconstruction loss
        main_loss = self.criterion(decoded_from_estimator, targets)
        
        # Check for NaN loss
        if torch.isnan(main_loss):
            print("WARNING: NaN loss detected!")
            main_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Training step
        if train:
            # Enable all gradients for end-to-end training
            self._set_requires_grad(self.model.generator, True)
            self._set_requires_grad(self.model.estimator, True)
            self._set_requires_grad(self.model.decoder, True)

            # Use the GE optimizer for end-to-end training
            self.optimizer_ge.zero_grad()
            main_loss.backward()
            
            # More aggressive gradient clipping for stability
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.model.generator.parameters()) + 
                list(self.model.estimator.parameters()) + 
                list(self.model.decoder.parameters()),
                max_norm=0.5,  # Reduced from 1.0 for stability
            )
            
            # Skip update if gradients are too large
            if grad_norm > 10.0:
                print(f"WARNING: Skipping update due to large gradients: {grad_norm:.2f}")
            else:
                self.optimizer_ge.step()
            
            # Debug: Print gradients occasionally
            if hasattr(self, '_debug_step') and self._debug_step % 50 == 0:
                total_grad_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
                print(f"Step {self._debug_step}: Main loss = {main_loss.item():.6f}, Grad norm = {total_grad_norm ** 0.5:.6f}")
                print(f"  Input range: [{inputs.min().item():.4f}, {inputs.max().item():.4f}]")
                print(f"  Target range: [{targets.min().item():.4f}, {targets.max().item():.4f}]")
                print(f"  Pred range: [{decoded_from_estimator.min().item():.4f}, {decoded_from_estimator.max().item():.4f}]")
                print(f"  GMF range: [{gmf_estimated.min().item():.4f}, {gmf_estimated.max().item():.4f}]")
            if not hasattr(self, '_debug_step'):
                self._debug_step = 0
            self._debug_step += 1

        # Compute additional metrics for logging (but don't use for training)
        with torch.no_grad():
            gmf_generated = self.model.generate_gmf(params, targets)
            decoded_from_generator = self.model.decode(params, gmf_generated)
            
            l1_val = self.criterion(gmf_estimated, gmf_generated).item()
            l2_val = self.criterion(decoded_from_generator, targets).item()

        # Compute final metrics
        preds_denorm = decoded_from_estimator * self.label_std_tensor + self.label_mean_tensor
        targets_denorm = targets * self.label_std_tensor + self.label_mean_tensor
        diff = preds_denorm - targets_denorm
        mse = torch.mean(diff ** 2).item()
        rmse = float(np.sqrt(mse))
        accuracy = self._compute_accuracy(decoded_from_estimator, targets)

        return {
            'loss': main_loss.item(),
            'l1': l1_val,
            'l2': l2_val,
            'rmse': rmse,
            'accuracy': accuracy,
        }

    def train_epoch(self, train_loader) -> Dict[str, float]:
        self.model.train()
        metrics = {'loss': 0.0, 'l1': 0.0, 'l2': 0.0, 'rmse': 0.0, 'accuracy': 0.0}
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
        metrics = {'loss': 0.0, 'l1': 0.0, 'l2': 0.0, 'rmse': 0.0, 'accuracy': 0.0}
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
        metrics = {'loss': 0.0, 'rmse': 0.0, 'mae': 0.0, 'accuracy': 0.0}
        batches = 0

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    inputs, targets, params = batch
                else:
                    inputs, targets = batch
                    # Create dummy parameters if not available (mass=70kg, height=1.7m as defaults)
                    params = torch.tensor([[70.0, 1.7]] * targets.shape[0], device=self.device)
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                params = params.to(self.device)
                
                # Normalize parameters if available
                if self.param_mean_tensor is not None and self.param_std_tensor is not None:
                    params = (params - self.param_mean_tensor) / self.param_std_tensor

                gmf_estimated = self.model.estimator(inputs)
                decoded = self.model.decode(params, gmf_estimated)

                preds_denorm = decoded * self.label_std_tensor + self.label_mean_tensor
                targets_denorm = targets * self.label_std_tensor + self.label_mean_tensor

                diff = preds_denorm - targets_denorm
                mse = torch.mean(diff ** 2).item()
                mae = torch.mean(torch.abs(diff)).item()

                metrics['loss'] += mse
                metrics['rmse'] += np.sqrt(mse)
                metrics['mae'] += mae
                metrics['accuracy'] += self._compute_accuracy(decoded, targets)
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
            self.current_epoch = epoch
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.eval_epoch(val_loader)

            self.train_accuracy_history.append(train_metrics['accuracy'])
            self.val_accuracy_history.append(val_metrics['accuracy'])
            self.train_rmse_history.append(train_metrics['rmse'])
            self.val_rmse_history.append(val_metrics['rmse'])

            if self.run is not None:
                log_dict = {
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/rmse': train_metrics['rmse'],
                    'train/accuracy': train_metrics['accuracy'],
                    'train/L1': train_metrics['l1'],
                    'train/L2': train_metrics['l2'],
                    'val/loss': val_metrics['loss'],
                    'val/rmse': val_metrics['rmse'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/L1': val_metrics['l1'],
                    'val/L2': val_metrics['l2'],
                    'lr/ge': self.optimizer_ge.param_groups[0]['lr'],
                }
                if self.optimizer_gd is not None:
                    log_dict['lr/gd'] = self.optimizer_gd.param_groups[0]['lr']
                wandb.log(log_dict)

            # Use validation RMSE for learning rate scheduling (more stable than loss)
            if self.scheduler_ge is not None:
                self.scheduler_ge.step(val_metrics['rmse'])
            if self.scheduler_gd is not None:
                self.scheduler_gd.step(val_metrics['rmse'])

            if val_metrics['rmse'] < self.best_val_loss:
                self.best_val_loss = val_metrics['rmse']
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch)
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (patience: {self.early_stopping_patience})")
                break

        self._save_accuracy_plot()

    def _save_accuracy_plot(self) -> None:
        if not self.train_accuracy_history:
            return

        epochs = list(range(1, len(self.train_accuracy_history) + 1))

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_accuracy_history, label='Train Accuracy', marker='o')
        plt.plot(epochs, self.val_accuracy_history, label='Validation Accuracy', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('GMF Training Accuracy')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(self.save_dir, 'accuracy_curve.png')
        plt.savefig(plot_path)
        plt.close()

        if self.run is not None:
            self.run.log({'accuracy_curve': wandb.Image(plot_path)})
