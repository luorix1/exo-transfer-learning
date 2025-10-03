import torch
from tqdm.auto import tqdm
import os
os.environ["MKL_VERBOSE"] = "0"
os.environ["MKL_DISABLE_FAST_MM"] = "1"
import wandb
import matplotlib.pyplot as plt
import numpy as np
import gc
import csv
from typing import Dict, Any


class Trainer:
    """Trainer class for TCN model training and evaluation."""
    
    def __init__(self, device: torch.device, model: torch.nn.Module, wandb_run, 
                 criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler, data_handler, 
                 config: Dict[str, Any], save_dir: str):
        
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_handler = data_handler
        self.hyperparam_config = config
        self.save_dir = save_dir
        self.run = wandb_run
        
        # Prepare mean and std tensors
        self.label_mean_tensor = torch.tensor(self.data_handler.label_mean, device=self.device)
        self.label_std_tensor = torch.tensor(self.data_handler.label_std, device=self.device)
        
        # For tracking best validation loss
        self.best_val_loss = float('inf')
        self.patience = 10  # early stopping
        self.patience_counter = 0
        
        # For tracking RMSE
        self.train_rmse_list = []
        self.val_rmse_list = []
        
        # Save model architecture
        model_arch = str(model)
        arch_file_path = os.path.join(self.save_dir, 'model_architecture.txt')
        with open(arch_file_path, "w") as arch_file:
            arch_file.write(model_arch)
        
        # Log the file as an artifact if wandb is available
        if self.run is not None:
            artifact = wandb.Artifact('model_architecture', type='model')
            artifact.add_file(arch_file_path)
            self.run.log_artifact(artifact)
        
        # Store the transfer_learning flag
        self.transfer_learning = config['transfer_learning']
        
    def compute_accuracy(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute accuracy based on threshold."""
        # Denormalize predictions and targets
        preds_denorm = preds * self.label_std_tensor.to(self.device) + self.label_mean_tensor.to(self.device)
        targets_denorm = targets * self.label_std_tensor.to(self.device) + self.label_mean_tensor.to(self.device)

        # Compute absolute error in N-m/kg
        absolute_error = torch.abs(preds_denorm - targets_denorm)

        # Determine which predictions are within the threshold
        threshold = 0.05  # N-m/kg for moment
        accurate_predictions = (absolute_error <= threshold).float()

        # Compute the mean accuracy across all outputs and samples
        accuracy = accurate_predictions.mean().item() * 100  # Convert to percentage
        return accuracy
        
    def train_epoch(self, train_loader) -> tuple:
        """Train for one epoch."""
        self.model.train()
        tloss = 0
        trmse = 0
        tacc = 0
        total_samples = 0
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

        for i, (input, label) in enumerate(train_loader):
            self.optimizer.zero_grad()
            input = input.to(self.device)
            label = label.to(self.device)
            
            # Check for NaN/Inf in input data
            if torch.isnan(input).any() or torch.isinf(input).any():
                print(f"\n⚠️ WARNING: NaN/Inf detected in input at batch {i}")
                continue
            if torch.isnan(label).any() or torch.isinf(label).any():
                print(f"\n⚠️ WARNING: NaN/Inf detected in label at batch {i}")
                continue
            
            logits = self.model(input)
            
            # Check for NaN/Inf in model output
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"\n⚠️ WARNING: NaN/Inf in model output at batch {i}")
                print(f"   Input range: [{input.min().item():.4f}, {input.max().item():.4f}]")
                print(f"   Checking model parameters for NaN/Inf...")
                
                nan_params = []
                for name, param in self.model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        nan_params.append(name)
                
                if nan_params:
                    print(f"   ❌ Found NaN/Inf in parameters: {nan_params[:5]}")  # Show first 5
                    print(f"   Model has been corrupted - training should stop!")
                    raise RuntimeError(f"Model parameters contain NaN/Inf at batch {i}")
                else:
                    print(f"   Model parameters are OK - NaN from forward pass calculation")
                    print(f"   Skipping this batch to prevent gradient corruption")
                continue
            
            loss = self.criterion(logits, label)
            
            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️ WARNING: NaN/Inf loss at batch {i}")
                print(f"   Output range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
                print(f"   Label range: [{label.min().item():.4f}, {label.max().item():.4f}]")
                continue
            
            loss.backward()
            
            # Clip gradients to prevent explosion
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Check if gradients are too large (sign of instability)
            if grad_norm > 10.0:
                print(f"\n⚠️ WARNING: Large gradient norm {grad_norm:.2f} at batch {i}")
            
            self.optimizer.step()
            tloss += loss.item()

            # Denormalize for RMSE calculation
            preds_denorm = logits * self.label_std_tensor + self.label_mean_tensor
            targets_denorm = label * self.label_std_tensor + self.label_mean_tensor
            mse = torch.mean((preds_denorm - targets_denorm) ** 2).item()
            rmse = np.sqrt(mse)
            trmse += rmse

            # Compute accuracy
            accuracy = self.compute_accuracy(logits, label)
            tacc += accuracy

            total_samples += 1

            batch_bar.set_postfix(
                loss="{:.04f}".format(tloss / total_samples),
                rmse="{:.04f}".format(trmse / total_samples),
                acc="{:.02f}%".format(tacc / total_samples)
            )
            batch_bar.update()
            del input, label, logits
            torch.cuda.empty_cache()

        batch_bar.close()
        tloss /= len(train_loader)
        trmse /= len(train_loader)
        tacc /= len(train_loader)
        return tloss, trmse, tacc
        
    def eval_epoch(self, val_loader) -> tuple:
        """Evaluate for one epoch."""
        self.model.eval()
        vloss = 0
        vrmse = 0
        vacc = 0
        total_samples = 0
        batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

        with torch.no_grad():
            for i, (input, label) in enumerate(val_loader):
                input = input.to(self.device)
                label = label.to(self.device)
                logits = self.model(input)
                loss = self.criterion(logits, label)
                vloss += loss.item()

                # Denormalize for RMSE calculation
                preds_denorm = logits * self.label_std_tensor + self.label_mean_tensor
                targets_denorm = label * self.label_std_tensor + self.label_mean_tensor
                mse = torch.mean((preds_denorm - targets_denorm) ** 2).item()
                rmse = np.sqrt(mse)
                vrmse += rmse

                # Compute accuracy
                accuracy = self.compute_accuracy(logits, label)
                vacc += accuracy

                total_samples += 1

                batch_bar.set_postfix(
                    loss="{:.04f}".format(vloss / total_samples),
                    rmse="{:.04f}".format(vrmse / total_samples),
                    acc="{:.02f}%".format(vacc / total_samples)
                )
                batch_bar.update()
                del input, label, logits
                torch.cuda.empty_cache()

        batch_bar.close()
        vloss /= len(val_loader)
        vrmse /= len(val_loader)
        vacc /= len(val_loader)
        return vloss, vrmse, vacc
        
    def plot_predictions(self, dataloader, num_samples: int = 1000, epoch: int = 0, prefix: str = ''):
        """Plot predictions vs ground truth with scatter plot and R² score."""
        self.model.eval()
        label_true_list = []
        label_pred_list = []
    
        with torch.no_grad():
            for input, label in dataloader:
                input = input.to(self.device)
                label = label.to(self.device)
                logits = self.model(input)
                # Denormalize
                preds_denorm = logits * self.label_std_tensor.to(self.device) + self.label_mean_tensor.to(self.device)
                targets_denorm = label * self.label_std_tensor.to(self.device) + self.label_mean_tensor.to(self.device)
                label_true_list.append(targets_denorm.cpu())
                label_pred_list.append(preds_denorm.cpu())
        
        label_true = torch.cat(label_true_list, dim=0)
        label_pred = torch.cat(label_pred_list, dim=0)
        
        # Remove NaN values
        label_true_np = label_true.numpy()
        label_pred_np = label_pred.numpy()
        
        # Handle case where output size might be different from actual data
        # Determine actual output size from tensors and configured side
        actual_output_size = min(label_true.shape[1], label_pred.shape[1], self.hyperparam_config['output_size'])
        for i in range(actual_output_size):
            # Get data for this joint
            true_i = label_true_np[:, i]
            pred_i = label_pred_np[:, i]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(true_i) | np.isnan(pred_i))
            true_i_clean = true_i[valid_mask]
            pred_i_clean = pred_i[valid_mask]
            
            if len(true_i_clean) == 0:
                continue
            
            # Calculate R² score
            ss_res = np.sum((true_i_clean - pred_i_clean) ** 2)
            ss_tot = np.sum((true_i_clean - np.mean(true_i_clean)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((true_i_clean - pred_i_clean) ** 2))
            
            # Create scatter plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.scatter(true_i_clean, pred_i_clean, alpha=0.3, s=5, color='blue', edgecolors='none')
            
            # Plot y=x line
            min_val = min(true_i_clean.min(), pred_i_clean.min())
            max_val = max(true_i_clean.max(), pred_i_clean.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
            
            # Add R² and RMSE to plot
            ax.text(0.05, 0.95, f'R² = {r2_score:.4f}\nRMSE = {rmse:.4f} N-m/kg', 
                    transform=ax.transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel('True Hip Moment (N-m/kg)', fontsize=11)
            ax.set_ylabel('Predicted Hip Moment (N-m/kg)', fontsize=11)
            ax.set_title(f'Epoch {epoch} - Joint {i} Predictions', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            filename = os.path.join(self.save_dir, f'prediction_epoch_{epoch}_joint_{i}.png')
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()

        # Save true and predicted labels to CSV
        csv_file = os.path.join(self.save_dir, f'predictions_epoch_{epoch}.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header - use actual output size
            header = [f'true_{i}' for i in range(actual_output_size)] + \
                 [f'pred_{i}' for i in range(actual_output_size)]
            writer.writerow(header)
            
            # Write data
            for i in range(label_true.shape[0]):
                row = label_true[i, :actual_output_size].numpy().tolist() + label_pred[i, :actual_output_size].numpy().tolist()
                writer.writerow(row)
    
    def train(self):
        """Main training loop."""
        torch.cuda.empty_cache()
        gc.collect()
        num_epochs = self.hyperparam_config['epochs']
        
        # Get train/val split and create dataloaders once at the beginning
        train_indices, val_indices = self.data_handler.get_train_val_indices()
        train_loader, val_loader = self.data_handler.create_dataloaders(train_indices, val_indices)
        test_loader = self.data_handler.create_dataloaders(test_indices=1)
        
        for epoch in range(num_epochs):
            print("\nEpoch {}/{}".format(epoch+1, num_epochs))
            curr_lr = float(self.optimizer.param_groups[0]['lr'])

            train_loss, train_rmse, train_acc = self.train_epoch(train_loader)
            val_loss, val_rmse, val_acc = self.eval_epoch(val_loader)
            
            self.scheduler.step(val_loss)
            print("\tTrain Loss {:.04f}\tRMSE {:.04f}\tLearning Rate {:.7f}".format(
                train_loss, train_rmse, curr_lr))
            print("\tVal Loss {:.04f}\t\tRMSE {:.04f}".format(
                val_loss, val_rmse))

            # Save RMSE values for plotting
            self.train_rmse_list.append(train_rmse)
            self.val_rmse_list.append(val_rmse)

            torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"{self.hyperparam_config['wandb_session_name']}_epoch_{epoch+1}.pt"))

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save the best model
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, self.hyperparam_config['wandb_session_name'] + '.pt'))
            else:
                self.patience_counter += 1

            # Plot predictions after each epoch
            self.plot_predictions(test_loader, num_samples=5000, epoch=epoch+1)

            test_loss, test_rmse, test_acc = self.eval_epoch(test_loader)
            print("\nTest Loss {:.04f}\tRMSE {:.04f}".format(
                test_loss, test_rmse))

            # Early Stopping
            if self.patience_counter >= self.patience:
                print("Early stopping triggered")
                # Load the best model
                self.model.load_state_dict(torch.load(os.path.join(self.save_dir, self.hyperparam_config['wandb_session_name'] + '.pt')))
                # Plot predictions with the best model
                self.plot_predictions(test_loader, num_samples=10000, epoch='best')
                break
            
            # Log metrics to wandb
            if self.run is not None:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'train/rmse': train_rmse,
                    'train/accuracy': train_acc,
                    'val/loss': val_loss,
                    'val/rmse': val_rmse,
                    'val/accuracy': val_acc,
                    'test/loss': test_loss,
                    'test/rmse': test_rmse,
                    'test/accuracy': test_acc,
                    'learning_rate': curr_lr,
                    'best_val_loss': self.best_val_loss,
                    'patience_counter': self.patience_counter,
                })
                
                # Create and log summary plots for better visualization
                # These will be updated each epoch in wandb UI
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                # Plot 1: Loss over epochs
                epochs_so_far = list(range(1, epoch + 2))
                # Track losses (need to store them)
                if not hasattr(self, 'train_loss_list'):
                    self.train_loss_list = []
                    self.val_loss_list = []
                    self.test_loss_list = []
                    self.train_acc_list = []
                    self.val_acc_list = []
                    self.test_acc_list = []
                
                self.train_loss_list.append(train_loss)
                self.val_loss_list.append(val_loss)
                self.test_loss_list.append(test_loss)
                self.train_acc_list.append(train_acc)
                self.val_acc_list.append(val_acc)
                self.test_acc_list.append(test_acc)
                
                axes[0].plot(epochs_so_far, self.train_loss_list, 'b-', label='Train', linewidth=2)
                axes[0].plot(epochs_so_far, self.val_loss_list, 'r-', label='Val', linewidth=2)
                axes[0].plot(epochs_so_far, self.test_loss_list, 'g-', label='Test', linewidth=2)
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Loss over Epochs')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Plot 2: RMSE over epochs
                axes[1].plot(epochs_so_far, self.train_rmse_list, 'b-', label='Train', linewidth=2)
                axes[1].plot(epochs_so_far, self.val_rmse_list, 'r-', label='Val', linewidth=2)
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('RMSE (N-m/kg)')
                axes[1].set_title('RMSE over Epochs')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                axes[1].set_ylim(bottom=0)
                
                # Plot 3: Accuracy over epochs
                axes[2].plot(epochs_so_far, self.train_acc_list, 'b-', label='Train', linewidth=2)
                axes[2].plot(epochs_so_far, self.val_acc_list, 'r-', label='Val', linewidth=2)
                axes[2].plot(epochs_so_far, self.test_acc_list, 'g-', label='Test', linewidth=2)
                axes[2].set_xlabel('Epoch')
                axes[2].set_ylabel('Accuracy (%)')
                axes[2].set_title('Accuracy over Epochs')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                axes[2].set_ylim([0, 100])
                
                plt.tight_layout()
                
                # Log the combined plot
                wandb.log({'training_progress': wandb.Image(fig)})
                plt.close(fig)
        
        # Return test_loader for use in final evaluation
        return test_loader
        
    def evaluate(self, test_loader=None):
        """Evaluate the model on test data."""
        # Load the best model
        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, self.hyperparam_config['wandb_session_name'] + '.pt')))
        
        # Create test_loader if not provided
        if test_loader is None:
            test_loader = self.data_handler.create_dataloaders(test_indices=1)
        
        # Evaluate on Test Data
        test_loss, test_rmse, test_acc = self.eval_epoch(test_loader)
        
        print("\nFinal Test Loss {:.04f}\tRMSE {:.04f}".format(
            test_loss, test_rmse))
        
        test_metrics_file = os.path.join(self.save_dir, 'test_metrics.csv')
        fieldnames = ['test_rmse']

        file_exists = os.path.isfile(test_metrics_file)

        with open(test_metrics_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'test_rmse': test_rmse,
            })
        
        # Plot predictions using the best model (larger sample)
        self.plot_predictions(test_loader, num_samples=10000, epoch='final')

        # Plot final RMSE over epochs
        plt.figure()
        plt.plot(range(1, len(self.train_rmse_list)+1), self.train_rmse_list, label='Training RMSE')
        plt.plot(range(1, len(self.val_rmse_list)+1), self.val_rmse_list, label='Validation RMSE')
        plt.ylim(bottom=0)
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Training and Validation RMSE over Epochs')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'final_rmse_plot.png'))
        plt.close()
        
        # Log final metrics and artifacts to wandb
        if self.run is not None:
            # Log final test metrics
            wandb.log({
                'final_test_loss': test_loss,
                'final_test_rmse': test_rmse,
                'final_test_accuracy': test_acc,
                'total_epochs': len(self.train_rmse_list),
            })
            
            # Create comprehensive final plots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            epochs = list(range(1, len(self.train_rmse_list) + 1))
            
            # Plot 1: Loss over epochs
            if hasattr(self, 'train_loss_list'):
                axes[0, 0].plot(epochs, self.train_loss_list, 'b-o', label='Train', linewidth=2, markersize=4)
                axes[0, 0].plot(epochs, self.val_loss_list, 'r-s', label='Val', linewidth=2, markersize=4)
                axes[0, 0].plot(epochs, self.test_loss_list, 'g-^', label='Test', linewidth=2, markersize=4)
                axes[0, 0].set_xlabel('Epoch', fontsize=12)
                axes[0, 0].set_ylabel('Loss', fontsize=12)
                axes[0, 0].set_title('Loss over Epochs', fontsize=14, fontweight='bold')
                axes[0, 0].legend(fontsize=10)
                axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: RMSE over epochs
            axes[0, 1].plot(epochs, self.train_rmse_list, 'b-o', label='Train', linewidth=2, markersize=4)
            axes[0, 1].plot(epochs, self.val_rmse_list, 'r-s', label='Val', linewidth=2, markersize=4)
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('RMSE (N-m/kg)', fontsize=12)
            axes[0, 1].set_title('RMSE over Epochs', fontsize=14, fontweight='bold')
            axes[0, 1].legend(fontsize=10)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(bottom=0)
            
            # Plot 3: Accuracy over epochs
            if hasattr(self, 'train_acc_list'):
                axes[1, 0].plot(epochs, self.train_acc_list, 'b-o', label='Train', linewidth=2, markersize=4)
                axes[1, 0].plot(epochs, self.val_acc_list, 'r-s', label='Val', linewidth=2, markersize=4)
                axes[1, 0].plot(epochs, self.test_acc_list, 'g-^', label='Test', linewidth=2, markersize=4)
                axes[1, 0].set_xlabel('Epoch', fontsize=12)
                axes[1, 0].set_ylabel('Accuracy (%)', fontsize=12)
                axes[1, 0].set_title('Accuracy over Epochs', fontsize=14, fontweight='bold')
                axes[1, 0].legend(fontsize=10)
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_ylim([0, 100])
            
            # Plot 4: Learning rate schedule
            if hasattr(self, 'lr_list'):
                axes[1, 1].plot(epochs, self.lr_list, 'k-o', linewidth=2, markersize=4)
            else:
                # If we didn't track lr, show a text summary instead
                axes[1, 1].axis('off')
                summary_text = f"Final Metrics:\n\n"
                summary_text += f"Test RMSE: {test_rmse:.4f} N-m/kg\n"
                summary_text += f"Test Loss: {test_loss:.4f}\n"
                summary_text += f"Test Accuracy: {test_acc:.2f}%\n\n"
                summary_text += f"Best Val Loss: {self.best_val_loss:.4f}\n"
                summary_text += f"Total Epochs: {len(self.train_rmse_list)}"
                axes[1, 1].text(0.5, 0.5, summary_text, 
                              horizontalalignment='center',
                              verticalalignment='center',
                              fontsize=12,
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                              transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Training Summary', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            final_metrics_path = os.path.join(self.save_dir, 'final_training_metrics.png')
            plt.savefig(final_metrics_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Log plots as artifacts
            wandb.log({
                'final_rmse_plot': wandb.Image(os.path.join(self.save_dir, 'final_rmse_plot.png')),
                'final_training_metrics': wandb.Image(final_metrics_path)
            })
            
            # Save model as wandb artifact
            model_artifact = wandb.Artifact(
                name=f"tcn_model_{self.hyperparam_config['wandb_session_name']}",
                type="model",
                description=f"TCN model trained for joint moment prediction. Final RMSE: {test_rmse:.4f}"
            )
            model_artifact.add_file(os.path.join(self.save_dir, self.hyperparam_config['wandb_session_name'] + '.pt'))
            model_artifact.add_file(os.path.join(self.save_dir, 'input_mean.npy'))
            model_artifact.add_file(os.path.join(self.save_dir, 'input_std.npy'))
            model_artifact.add_file(os.path.join(self.save_dir, 'label_mean.npy'))
            model_artifact.add_file(os.path.join(self.save_dir, 'label_std.npy'))
            model_artifact.add_file(os.path.join(self.save_dir, 'model_architecture.txt'))
            self.run.log_artifact(model_artifact)
            
            print(f"Model and artifacts saved to wandb: {self.run.url}")
