import torch
from tqdm.auto import tqdm
import os
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
            logits = self.model(input)
            loss = self.criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
        """Plot predictions vs ground truth."""
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
        
        # Plot the prediction & true plot for each epoch
        # Handle case where output size might be different from actual data
        # Determine actual output size from tensors and configured side
        actual_output_size = min(label_true.shape[1], label_pred.shape[1], self.hyperparam_config['output_size'])
        for i in range(actual_output_size):
            plt.figure()
            plt.plot(label_true[:, i], label='True')
            plt.plot(label_pred[:, i], label='Predicted')
            plt.xlabel('Data number')
            plt.ylabel('Joint moment (N-m/kg)')
            plt.legend()
            filename = os.path.join(self.save_dir, f'prediction_epoch_{epoch}_joint_{i}.png')
            plt.savefig(filename)
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
        
        for epoch in range(num_epochs):
            print("\nEpoch {}/{}".format(epoch+1, num_epochs))
            curr_lr = float(self.optimizer.param_groups[0]['lr'])

            train_indices, val_indices = self.data_handler.get_train_val_indices()
            train_loader, val_loader = self.data_handler.create_dataloaders(train_indices, val_indices)

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
            test_loader = self.data_handler.create_dataloaders(test_indices=1)
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
                    'train_loss': train_loss,
                    'train_rmse': train_rmse,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'val_accuracy': val_acc,
                    'learning_rate': curr_lr,
                    'test_loss': test_loss,
                    'test_rmse': test_rmse,
                    'test_accuracy': test_acc,
                    'best_val_loss': self.best_val_loss,
                    'patience_counter': self.patience_counter,
                })
        
    def evaluate(self):
        """Evaluate the model on test data."""
        # Load the best model
        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, self.hyperparam_config['wandb_session_name'] + '.pt')))
                
        # Evaluate on Test Data
        test_loader = self.data_handler.create_dataloaders(test_indices=1)
        test_loss, test_rmse, test_acc = self.eval_epoch(test_loader)
        
        print("\nTest Loss {:.04f}\tRMSE {:.04f}".format(
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
        
        # Optionally, plot predictions using the best model
        self.plot_predictions(test_loader, num_samples=5000, epoch='final')

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
            
            # Log plots as artifacts
            wandb.log({
                'final_rmse_plot': wandb.Image(os.path.join(self.save_dir, 'final_rmse_plot.png'))
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
