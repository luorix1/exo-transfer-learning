import os
import numpy as np
import pandas as pd
from scipy import signal
import torch
from torch.utils.data import Subset, Dataset
from typing import List, Optional, Tuple


class DataHandler:
    """Handles data loading and preprocessing for the Canonical dataset format."""
    
    def __init__(self, data_root: str, hyperparam_config: dict, pretrained_model_path: Optional[str] = None):
        self.data_root = data_root
        self.window_size = hyperparam_config["window_size"]
        self.batch_size = hyperparam_config["batch_size"]
        self.num_workers = hyperparam_config["number_of_workers"]
        self.validation_split = hyperparam_config["validation_split"]
        self.dataset_proportion = hyperparam_config["dataset_proportion"]
        self.transfer_learning = hyperparam_config["transfer_learning"]
        self.pretrained_model_path = pretrained_model_path
        self.side = hyperparam_config.get('side', 'all')  # 'right', 'left', 'all'
            
        # Placeholder for data
        self.train_data = None
        self.test_data = None

        # Initialize mean and std attributes
        self.input_mean = None
        self.input_std = None
        self.label_mean = None
        self.label_std = None

        # (For transfer learning) Load saved mean and std for normalization
        if self.pretrained_model_path is not None:
            self.input_mean = np.load(os.path.join(self.pretrained_model_path, 'input_mean.npy'))
            self.input_std = np.load(os.path.join(self.pretrained_model_path, 'input_std.npy'))
            self.label_mean = np.load(os.path.join(self.pretrained_model_path, 'label_mean.npy'))
            self.label_std = np.load(os.path.join(self.pretrained_model_path, 'label_std.npy'))
        
    def load_data(self, train_data_partition: List[str], train_data_condition: List[str], 
                  test_data_partition: List[str], test_data_partition_2: Optional[List[str]] = None, 
                  test_data_partition_3: Optional[List[str]] = None):
        """Load training and test data."""
        
        # Load training data (including data for validation split)
        print("\n...Loading training data...\n")        
        self.train_data = LoadData(
            root=self.data_root,
            partitions=train_data_partition,
            conditions=train_data_condition,
            window_size=self.window_size,
            data_type="train_data" if self.transfer_learning == False else "train_data_transfer_learning",
            dataset_proportion=self.dataset_proportion,
            input_mean=self.input_mean,
            input_std=self.input_std,
            label_mean=self.label_mean,
            label_std=self.label_std
        )
        
        # Retrieve mean and std from training data
        self.input_mean = self.train_data.input_mean
        self.input_std = self.train_data.input_std
        self.label_mean = self.train_data.label_mean
        self.label_std = self.train_data.label_std
        
        # Load test data using training mean and std
        print("\n...Loading test data...\n")        
        self.test_data = LoadData(
            root=self.data_root,
            partitions=test_data_partition,
            conditions=train_data_condition,  # Use the same conditions as training data
            window_size=self.window_size,
            data_type="test_data",
            dataset_proportion=self.dataset_proportion
        )
        # Replace the mean and std of test data with that of training data
        self.test_data.input_mean = self.input_mean
        self.test_data.input_std = self.input_std
        self.test_data.label_mean = self.label_mean
        self.test_data.label_std = self.label_std
        
    def save_mean_std(self, save_dir: str):
        """Save mean and std to .npy files."""
        np.save(os.path.join(save_dir, 'input_mean.npy'), self.input_mean)
        np.save(os.path.join(save_dir, 'input_std.npy'), self.input_std)
        np.save(os.path.join(save_dir, 'label_mean.npy'), self.label_mean)
        np.save(os.path.join(save_dir, 'label_std.npy'), self.label_std)
     
    def get_train_val_indices(self) -> Tuple[List[int], List[int]]:
        """Randomly split indices for training and validation."""
        if len(self.train_data.subject_data_length) == 1:
            # Special case: only one subject
            total_length = self.train_data.subject_data_length[0] - self.window_size + 1
            indices = list(range(total_length))
            split_point = int(np.floor(total_length * (1 - self.validation_split)))
            
            # Shuffle indices before splitting
            np.random.shuffle(indices)
            train_indices = indices[:split_point]
            val_indices = indices[split_point:]
            print(f"\nSingle subject detected. Random split: {len(train_indices)} train, {len(val_indices)} validation samples")
        else:
            # Multiple subjects: leave one subject out
            leave_one_subject_out = np.random.randint(0, len(self.train_data.subject_data_length))
            print(f"\nLeave out subject: {leave_one_subject_out+1}")

            leave_out_start = sum(self.train_data.subject_data_length[:leave_one_subject_out])
            leave_out_end = sum(self.train_data.subject_data_length[:leave_one_subject_out+1]) - self.window_size + 1
            total_length = sum(self.train_data.subject_data_length) - self.window_size + 1
            train_indices = list(range(0, leave_out_start)) + list(range(leave_out_end, total_length))
            val_indices = list(range(leave_out_start, leave_out_end))

        return train_indices, val_indices
    
    def create_dataloaders(self, train_indices: Optional[List[int]] = None, 
                         val_indices: Optional[List[int]] = None, 
                         test_indices: Optional[int] = None):
        """Create DataLoaders for training, validation, or testing."""
        if train_indices is not None and val_indices is not None:
            train_subset = Subset(self.train_data, train_indices)
            val_subset = Subset(self.train_data, val_indices)
            
            train_loader = torch.utils.data.DataLoader(
                dataset=train_subset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                pin_memory=True,
                shuffle=True
            )
            
            val_loader = torch.utils.data.DataLoader(
                dataset=val_subset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                pin_memory=True,
                shuffle=False
            )
            return train_loader, val_loader
        else:
            if test_indices == 1:
                # Create Test DataLoader
                test_loader = torch.utils.data.DataLoader(
                    dataset=self.test_data,
                    num_workers=self.num_workers,
                    batch_size=self.batch_size,
                    pin_memory=True,
                    shuffle=False
                )
            
            return test_loader


class LoadData(Dataset):
    """Dataset class to load data from Canonical format."""
    
    def __init__(self, root: str, partitions: List[str], conditions: List[str], 
                 window_size: int, data_type: str, dataset_proportion: Optional[float] = None,
                 input_mean: Optional[np.ndarray] = None, input_std: Optional[np.ndarray] = None,
                 label_mean: Optional[np.ndarray] = None, label_std: Optional[np.ndarray] = None,
                 normalize: bool = True):
        self.window_size = window_size
        self.input_list = []
        self.label_list = []
        self.data_type = data_type
        self.normalize = normalize
        self.dataset_proportion = dataset_proportion
        self.subject_data_length = []

        for subject_num, subject in enumerate(partitions):  # Multiple partitions
            self.subject_data_length.append(0)  # Append 0 for each subject
            subject_path = os.path.join(root, subject)
            if not os.path.isdir(subject_path): 
                continue  # Skip if not a directory

            for condition in conditions:
                condition_path = os.path.join(subject_path, condition)
                if not os.path.isdir(condition_path): 
                    continue  # Skip if not a directory

                for trial in os.listdir(condition_path):
                    trial_path = os.path.join(condition_path, trial)
                    if not os.path.isdir(trial_path): 
                        continue  # Skip if not a directory

                    input_file_dir = os.path.join(trial_path, 'Input')
                    label_file_dir = os.path.join(trial_path, 'Label')
                    
                    if not os.path.exists(input_file_dir) or not os.path.exists(label_file_dir):
                        continue
                        
                    input_file_names = sorted(os.listdir(input_file_dir))
                    label_file_names = sorted(os.listdir(label_file_dir))

                    # Load and concatenate all input data files
                    input_buffer_R = None
                    input_buffer_L = None

                    for i, name in enumerate(input_file_names):
                        csv_path = os.path.join(input_file_dir, name)

                        # Extract IMU data from input file
                        if 'imu' in name.lower():
                            # Read IMU data with flexible delimiter
                            try:
                                df = pd.read_csv(csv_path, sep=None, engine='python', on_bad_lines='skip')
                            except:
                                df = pd.read_csv(csv_path, sep=',', on_bad_lines='skip')
                            
                            # Extract gyroscope data (canonical format has normalized column names)
                            gyro_cols = [col for col in df.columns if 'gyro' in col.lower()]
                            if len(gyro_cols) >= 6:  # Need at least 6 gyro channels (pelvis, thigh_r, thigh_l)
                                # Right side: pelvis + thigh_r gyro
                                pelvis_gyro = [col for col in gyro_cols if 'pelvis' in col.lower()]
                                thigh_r_gyro = [col for col in gyro_cols if 'thigh_r' in col.lower() or 'femur_r' in col.lower()]
                                
                                if pelvis_gyro and thigh_r_gyro:
                                    right_cols = pelvis_gyro + thigh_r_gyro
                                    input_df_R = df[right_cols].values
                                    
                                    # Left side: pelvis + thigh_l gyro (flip signs for left)
                                    thigh_l_gyro = [col for col in gyro_cols if 'thigh_l' in col.lower() or 'femur_l' in col.lower()]
                                    if thigh_l_gyro:
                                        left_cols = pelvis_gyro + thigh_l_gyro
                                        input_df_L = df[left_cols].values
                                        # Flip signs for left side gyro (Y and Z axes)
                                        input_df_L[:, [1, 3, 5]] *= -1
                                    else:
                                        # If no left thigh data, use right side data flipped
                                        input_df_L = input_df_R.copy()
                                        input_df_L[:, [1, 3, 5]] *= -1
                                else:
                                    continue
                            else:
                                continue

                        else:
                            continue

                        # Horizontally stack all input data
                        if input_buffer_R is None:
                            input_buffer_R = input_df_R
                        else:
                            input_buffer_R = np.hstack((input_buffer_R, input_df_R))
                        if input_buffer_L is None:
                            input_buffer_L = input_df_L
                        else:
                            input_buffer_L = np.hstack((input_buffer_L, input_df_L))

                    if input_buffer_R is None or input_buffer_L is None:
                        continue

                    # Segment train data and test data based on dataset_proportion
                    input_time_sec = int(input_buffer_R.shape[0]/100)  # Extract recording time from input file by dividing 100 Hz
                    
                    if self.data_type == "train_data":
                        input_buffer_R = input_buffer_R[:int(input_buffer_R.shape[0]* self.dataset_proportion), :]
                        input_buffer_L = input_buffer_L[:int(input_buffer_L.shape[0]* self.dataset_proportion), :]
                    elif self.data_type == "train_data_transfer_learning":
                        input_buffer_R = input_buffer_R[:int(input_buffer_R.shape[0]* self.dataset_proportion), :]
                        input_buffer_L = input_buffer_L[:int(input_buffer_L.shape[0]* self.dataset_proportion), :]
                    elif self.data_type == "test_data":
                        input_buffer_R = input_buffer_R[:int(input_buffer_R.shape[0]* self.dataset_proportion), :]
                        input_buffer_L = input_buffer_L[:int(input_buffer_L.shape[0]* self.dataset_proportion), :]
                    
                    # Build input based on requested side
                    if self.side == 'right':
                        input_buffer = input_buffer_R
                    elif self.side == 'left':
                        input_buffer = input_buffer_L
                    else:
                        # both sides stacked (R first then L)
                        input_buffer = np.vstack((input_buffer_R, input_buffer_L))

                    self.input_list.append(input_buffer)
                    self.subject_data_length[subject_num] += input_buffer.shape[0]
                    
                    # Load and label data file (joint moments)
                    if label_file_names:
                        label_file_path = os.path.join(label_file_dir, label_file_names[0])
                        try:
                            label_df = pd.read_csv(label_file_path, sep=None, engine='python', on_bad_lines='skip')
                        except:
                            label_df = pd.read_csv(label_file_path, sep=',', on_bad_lines='skip')
                        
                        # Extract hip moments (right and left)
                        hip_moment_cols = [col for col in label_df.columns if 'hip' in col.lower() and 'moment' in col.lower()]
                        
                        if len(hip_moment_cols) >= 1:
                            # Find right and left hip moment columns
                            hip_r_col = [col for col in hip_moment_cols if 'r' in col.lower() or 'right' in col.lower()]
                            hip_l_col = [col for col in hip_moment_cols if 'l' in col.lower() or 'left' in col.lower()]
                            
                            # Handle different cases: both sides, only right, only left, or single moment
                            if hip_r_col and hip_l_col:
                                # Both sides available
                                label_buffer = label_df[hip_r_col + hip_l_col].values
                            elif hip_r_col and not hip_l_col:
                                # Only right side - duplicate for left
                                label_buffer = np.column_stack([label_df[hip_r_col].values, label_df[hip_r_col].values])
                            elif hip_l_col and not hip_r_col:
                                # Only left side - duplicate for right
                                label_buffer = np.column_stack([label_df[hip_l_col].values, label_df[hip_l_col].values])
                            elif len(hip_moment_cols) == 1:
                                # Single moment column - assume it's right side, duplicate for left
                                label_buffer = np.column_stack([label_df[hip_moment_cols[0]].values, label_df[hip_moment_cols[0]].values])
                            else:
                                continue
                            
                            # Segment labels based on dataset_proportion
                            if self.data_type == "train_data":
                                label_buffer = label_buffer[:int(label_buffer.shape[0]* self.dataset_proportion), :]
                            elif self.data_type == "train_data_transfer_learning":
                                label_buffer = label_buffer[:int(label_buffer.shape[0]* self.dataset_proportion), :]
                            elif self.data_type == "test_data":
                                label_buffer = label_buffer[:int(label_buffer.shape[0]* self.dataset_proportion), :]
                            
                            # Build labels based on requested side
                            if self.side == 'right':
                                label_buffer = label_buffer[:, 0:1]
                            elif self.side == 'left':
                                label_buffer = label_buffer[:, 1:2]
                            else:
                                # both sides
                                label_buffer = label_buffer[:, 0:2]
                            
                            self.label_list.append(label_buffer)
                        else:
                            continue
                    else:
                        continue
            
            print(f"{subject} data length: {self.subject_data_length[subject_num]}")

        if not self.input_list or not self.label_list:
            raise ValueError("No valid data found in the specified partitions and conditions")

        # Concatenate all data
        self.input = np.concatenate(self.input_list, axis=0)
        self.label = np.concatenate(self.label_list, axis=0)

        print(f'\ninput {self.data_type} dataset size: {np.shape(self.input)}')
        print(f'label {self.data_type} dataset size: {np.shape(self.label)}')

        self.length = len(self.input) - self.window_size + 1
        print(f"Total {self.data_type} sequences: {self.length}")

        # Calculate mean and std using the entire dataset
        self.input_mean = np.mean(self.input, axis=0)
        self.input_std = np.std(self.input, axis=0) + 1e-8

        self.label_mean = np.mean(self.label, axis=0)
        self.label_std = np.std(self.label, axis=0) + 1e-8

        # Override with provided mean and std if given (for transfer learning)
        if input_mean is not None:
            self.input_mean = input_mean
        if input_std is not None:
            self.input_std = input_std
        if label_mean is not None:
            self.label_mean = label_mean
        if label_std is not None:
            self.label_std = label_std

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        windows_input = self.input[ind: ind + self.window_size]  # Shape: (window_size, input_size)
        # Normalize the data using stored mean and std
        if self.normalize == True:
            windows_input = (windows_input - self.input_mean) / self.input_std

        # Convert to tensor without flattening
        window_input = torch.FloatTensor(windows_input).T  # Shape: (input_size, window_size)

        # Get the target joint moments at the last time point in the window
        target_label = self.label[ind + self.window_size - 1]  # Shape: (output_size)
        
        # Normalize the target joint moments
        if self.normalize == True:
            target_label = (target_label - self.label_mean) / self.label_std
            
        window_label = torch.FloatTensor(target_label)  # Shape: (output_size)
        
        return window_input, window_label
