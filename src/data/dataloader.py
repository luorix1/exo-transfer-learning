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
        self.imu_segments = hyperparam_config.get("imu_segments", ["pelvis", "femur"])
        self.augment = hyperparam_config.get("augment", False)
        self.label_filter_hz = float(hyperparam_config.get("label_filter_hz", 6.0))
        self.normalize = hyperparam_config.get("normalize", True)
            
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
            label_std=self.label_std,
            normalize=self.normalize,
            imu_segments=self.imu_segments,
            augment=self.augment,
            label_filter_hz=self.label_filter_hz
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
            dataset_proportion=self.dataset_proportion,
            normalize=self.normalize,
            imu_segments=self.imu_segments,
            augment=False,
            label_filter_hz=self.label_filter_hz
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
                shuffle=True
            )
            
            val_loader = torch.utils.data.DataLoader(
                dataset=val_subset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
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
                    shuffle=False
                )
            
            return test_loader


class LoadData(Dataset):
    """Dataset class to load data from Canonical format."""
    
    def __init__(self, root: str, partitions: List[str], conditions: List[str], 
                 window_size: int, data_type: str, dataset_proportion: Optional[float] = None,
                 input_mean: Optional[np.ndarray] = None, input_std: Optional[np.ndarray] = None,
                 label_mean: Optional[np.ndarray] = None, label_std: Optional[np.ndarray] = None,
                 normalize: bool = True, imu_segments: Optional[List[str]] = None,
                 augment: bool = False,
                 label_filter_hz: float = 6.0):
        self.window_size = window_size
        self.input_list = []
        self.label_list = []
        self.data_type = data_type
        self.normalize = normalize
        self.dataset_proportion = dataset_proportion
        self.subject_data_length = []
        # Default to femur+pelvis if not specified
        self.imu_segments = imu_segments if imu_segments is not None else ["pelvis", "femur"]
        self.augment = augment and (data_type.startswith("train"))
        self.label_filter_hz = float(label_filter_hz)

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
                            
                        # Extract gyroscope data based on configured IMU segments
                        gyro_cols = [col for col in df.columns if 'gyro' in col.lower()]
                        
                        right_cols = []
                        left_cols = []
                        
                        # Configure based on imu_segments parameter
                        if len(self.imu_segments) == 1 and self.imu_segments[0].lower() in ['femur', 'thigh']:
                            # Single femur/thigh IMU mode (3 channels)
                            thigh_r_gyro = [col for col in gyro_cols if 'thigh_r' in col.lower() or 'femur_r' in col.lower()]
                            thigh_l_gyro = [col for col in gyro_cols if 'thigh_l' in col.lower() or 'femur_l' in col.lower()]
                            
                            if not thigh_r_gyro or len(thigh_r_gyro) < 3:
                                raise ValueError(
                                    f"Required IMU segment 'femur/thigh' not found in {csv_path}\n"
                                    f"Available gyro columns: {gyro_cols}\n"
                                    f"Make sure your data contains 'thigh_r' or 'femur_r' gyro data."
                                )
                            
                            right_cols = thigh_r_gyro[:3]
                            # If left exists, use it; otherwise set to empty (will be None later)
                            if thigh_l_gyro and len(thigh_l_gyro) >= 3:
                                left_cols = thigh_l_gyro[:3]
                        
                        elif len(self.imu_segments) == 2:
                            # Dual IMU mode - check which segments are requested
                            seg1 = self.imu_segments[0].lower()
                            seg2 = self.imu_segments[1].lower()
                            
                            # Find pelvis gyro
                            pelvis_gyro = [col for col in gyro_cols if 'pelvis' in col.lower()]
                            # Find femur/thigh gyro
                            thigh_r_gyro = [col for col in gyro_cols if 'thigh_r' in col.lower() or 'femur_r' in col.lower()]
                            thigh_l_gyro = [col for col in gyro_cols if 'thigh_l' in col.lower() or 'femur_l' in col.lower()]
                            
                            # Verify we have the required segments
                            if 'pelvis' in [seg1, seg2] and ('femur' in [seg1, seg2] or 'thigh' in [seg1, seg2]):
                                if not pelvis_gyro or len(pelvis_gyro) < 3:
                                    raise ValueError(
                                        f"Required IMU segment 'pelvis' not found in {csv_path}\n"
                                        f"Available gyro columns: {gyro_cols}\n"
                                        f"Requested segments: {self.imu_segments}"
                                    )
                                if not thigh_r_gyro or len(thigh_r_gyro) < 3:
                                    raise ValueError(
                                        f"Required IMU segment 'femur/thigh' not found in {csv_path}\n"
                                        f"Available gyro columns: {gyro_cols}\n"
                                        f"Requested segments: {self.imu_segments}"
                                    )
                                
                                right_cols = pelvis_gyro[:3] + thigh_r_gyro[:3]
                                if thigh_l_gyro and len(thigh_l_gyro) >= 3:
                                    left_cols = pelvis_gyro[:3] + thigh_l_gyro[:3]
                            else:
                                raise ValueError(
                                    f"Unsupported IMU segment configuration: {self.imu_segments}\n"
                                    f"Supported configurations:\n"
                                    f"  - ['femur'] or ['thigh'] for single IMU\n"
                                    f"  - ['pelvis', 'femur'] or ['pelvis', 'thigh'] for dual IMU"
                                )
                        else:
                            raise ValueError(
                                f"Invalid number of IMU segments: {len(self.imu_segments)}\n"
                                f"Supported: 1 segment ['femur'] or 2 segments ['pelvis', 'femur']"
                            )

                        if not right_cols:
                            raise ValueError(
                                f"Failed to extract right side IMU data from {csv_path}\n"
                                f"Requested segments: {self.imu_segments}\n"
                                f"Available gyro columns: {gyro_cols}"
                            )

                        input_df_R = df[right_cols].values
                        if left_cols:
                            input_df_L = df[left_cols].values
                            # Flip signs for left side gyro (Y-axis for canonical frame)
                            if input_df_L.shape[1] >= 6:
                                input_df_L[:, [1, 4]] *= -1  # Y-axes of both sensors
                            elif input_df_L.shape[1] >= 3:
                                input_df_L[:, 1] *= -1  # Y-axis only
                        else:
                            # If no left columns, set to None - we'll only use right side data
                            input_df_L = None

                        # Horizontally stack all input data
                        if input_buffer_R is None:
                            input_buffer_R = input_df_R
                        else:
                            input_buffer_R = np.hstack((input_buffer_R, input_df_R))
                        
                        # Only stack left if it exists
                        if input_df_L is not None:
                            if input_buffer_L is None:
                                input_buffer_L = input_df_L
                            else:
                                input_buffer_L = np.hstack((input_buffer_L, input_df_L))

                    # Must have at least right side data
                    if input_buffer_R is None:
                        continue

                    # Load and label data file (joint moments)
                    if not label_file_names:
                        continue
                        
                    label_file_path = os.path.join(label_file_dir, label_file_names[0])
                    try:
                        label_df = pd.read_csv(label_file_path, sep=None, engine='python', on_bad_lines='skip')
                    except:
                        label_df = pd.read_csv(label_file_path, sep=',', on_bad_lines='skip')
                    
                    # Extract hip flexion moments (right and left)
                    # Look for hip_flexion_r_moment and hip_flexion_l_moment
                    hip_flexion_r_col = [col for col in label_df.columns 
                                        if 'hip_flexion_r_moment' in col.lower()]
                    hip_flexion_l_col = [col for col in label_df.columns 
                                        if 'hip_flexion_l_moment' in col.lower()]
                    
                    # Handle different cases: both sides or single moment
                    label_buffer_R = None
                    label_buffer_L = None
                    
                    if hip_flexion_r_col:
                        label_buffer_R = label_df[hip_flexion_r_col[0]].values.reshape(-1, 1)
                    if hip_flexion_l_col:
                        label_buffer_L = label_df[hip_flexion_l_col[0]].values.reshape(-1, 1)
                    
                    # If no hip flexion moments found, skip this trial
                    if label_buffer_R is None and label_buffer_L is None:
                        # Fallback: try generic hip moment columns
                        hip_moment_cols = [col for col in label_df.columns 
                                         if 'hip' in col.lower() and 'moment' in col.lower()]
                        if hip_moment_cols:
                            hip_r_col = [col for col in hip_moment_cols 
                                       if 'r' in col.lower() or 'right' in col.lower()]
                            hip_l_col = [col for col in hip_moment_cols 
                                       if 'l' in col.lower() or 'left' in col.lower()]
                            
                            if hip_r_col:
                                label_buffer_R = label_df[hip_r_col[0]].values.reshape(-1, 1)
                            if hip_l_col:
                                label_buffer_L = label_df[hip_l_col[0]].values.reshape(-1, 1)
                        
                        if label_buffer_R is None and label_buffer_L is None:
                            continue
                    
                    # Apply zero-phase Butterworth low-pass filtering to label(s) at 6 Hz (Fs=100 Hz)
                    def butter_lowpass_zero_phase(data: np.ndarray, cutoff_hz: float = None, fs_hz: float = 100.0, order: int = 4) -> np.ndarray:
                        if data is None:
                            return None
                        if data.size == 0:
                            return data
                        # Design Butterworth
                        nyq = 0.5 * fs_hz
                        cutoff = self.label_filter_hz if cutoff_hz is None else cutoff_hz
                        wn = float(cutoff) / nyq
                        b, a = signal.butter(order, wn, btype='low', analog=False)
                        # filtfilt along time axis (axis=0). Expect shape (N, 1)
                        try:
                            return signal.filtfilt(b, a, data.squeeze(), axis=0, method='pad', padlen=min(3 * max(len(a), len(b)), max(0, len(data) - 1))).reshape(-1, 1)
                        except ValueError:
                            # If sequence too short for padlen, fall back to lfilter twice
                            y = signal.lfilter(b, a, data.squeeze(), axis=0)
                            y = signal.lfilter(b, a, y[::-1], axis=0)[::-1]
                            return y.reshape(-1, 1)

                    label_buffer_R = butter_lowpass_zero_phase(label_buffer_R)
                    label_buffer_L = butter_lowpass_zero_phase(label_buffer_L)

                    # Process right side if we have both input and label
                    if input_buffer_R is not None and label_buffer_R is not None:
                        # Ensure input and label lengths match
                        min_len_R = min(input_buffer_R.shape[0], label_buffer_R.shape[0])
                        input_buffer_R = input_buffer_R[:min_len_R]
                        label_buffer_R = label_buffer_R[:min_len_R]
                        
                        # Filter valid rows for RIGHT side
                        valid_mask_R = (~np.isnan(input_buffer_R).any(axis=1)) & (~np.isnan(label_buffer_R).any(axis=1))
                        input_buffer_R_clean = input_buffer_R[valid_mask_R]
                        label_buffer_R_clean = label_buffer_R[valid_mask_R]
                    else:
                        input_buffer_R_clean = None
                        label_buffer_R_clean = None
                    
                    # Process left side if we have both input and label
                    if input_buffer_L is not None and label_buffer_L is not None:
                        # Ensure input and label lengths match
                        min_len_L = min(input_buffer_L.shape[0], label_buffer_L.shape[0])
                        input_buffer_L = input_buffer_L[:min_len_L]
                        label_buffer_L = label_buffer_L[:min_len_L]
                        
                        # Filter valid rows for LEFT side
                        valid_mask_L = (~np.isnan(input_buffer_L).any(axis=1)) & (~np.isnan(label_buffer_L).any(axis=1))
                        input_buffer_L_clean = input_buffer_L[valid_mask_L]
                        label_buffer_L_clean = label_buffer_L[valid_mask_L]
                    else:
                        input_buffer_L_clean = None
                        label_buffer_L_clean = None
                    
                    # Optional: training-time augmentation (rotations, noise, time stretch)
                    def apply_augmentation(arr: np.ndarray) -> np.ndarray:
                        if not self.augment:
                            return arr
                        if arr is None or len(arr) == 0:
                            return arr
                        data = arr.copy()
                        # Small random rotation per sample around axes for gyro triplets (per sensor)
                        def rotate_triples(mat: np.ndarray) -> np.ndarray:
                            m = mat.copy()
                            # Determine number of sensors (3 channels per sensor)
                            num_channels = m.shape[1]
                            if num_channels % 3 != 0:
                                return m
                            num_sensors = num_channels // 3
                            for s in range(num_sensors):
                                sl = slice(3*s, 3*s+3)
                                gx = np.deg2rad(np.random.uniform(-10, 10))
                                gy = np.deg2rad(np.random.uniform(-10, 10))
                                gz = np.deg2rad(np.random.uniform(-10, 10))
                                Rx = np.array([[1,0,0],[0,np.cos(gx),-np.sin(gx)],[0,np.sin(gx),np.cos(gx)]])
                                Ry = np.array([[np.cos(gy),0,np.sin(gy)],[0,1,0],[-np.sin(gy),0,np.cos(gy)]])
                                Rz = np.array([[np.cos(gz),-np.sin(gz),0],[np.sin(gz),np.cos(gz),0],[0,0,1]])
                                R = Rz @ Ry @ Rx
                                m[:, sl] = m[:, sl] @ R.T
                            return m
                        data = rotate_triples(data)
                        # Add Gaussian noise
                        data = data + np.random.normal(scale=0.01, size=data.shape)
                        return data

                    input_buffer_R_clean = apply_augmentation(input_buffer_R_clean)
                    if input_buffer_L_clean is not None:
                        input_buffer_L_clean = apply_augmentation(input_buffer_L_clean)

                    # Stack available data (unilateral training)
                    if input_buffer_R_clean is not None and len(input_buffer_R_clean) > 0:
                        if input_buffer_L_clean is not None and len(input_buffer_L_clean) > 0:
                            # Both sides available - randomly stack
                            R_side_first = np.random.randint(0, 2)
                            if R_side_first:
                                input_buffer = np.vstack((input_buffer_R_clean, input_buffer_L_clean))
                                label_buffer = np.vstack((label_buffer_R_clean, label_buffer_L_clean))
                            else:
                                input_buffer = np.vstack((input_buffer_L_clean, input_buffer_R_clean))
                                label_buffer = np.vstack((label_buffer_L_clean, label_buffer_R_clean))
                        else:
                            # Only right side available
                            input_buffer = input_buffer_R_clean
                            label_buffer = label_buffer_R_clean
                    elif input_buffer_L_clean is not None and len(input_buffer_L_clean) > 0:
                        # Only left side available
                        input_buffer = input_buffer_L_clean
                        label_buffer = label_buffer_L_clean
                    else:
                        # No valid data
                        continue

                    self.input_list.append(input_buffer)
                    self.label_list.append(label_buffer)
                    self.subject_data_length[subject_num] += input_buffer.shape[0]
            
            print(f"{subject} data length: {self.subject_data_length[subject_num]}")

        if not self.input_list or not self.label_list:
            raise ValueError("No valid data found in the specified partitions and conditions")

        # Concatenate all data (matching reference implementation)
        self.input = np.concatenate(self.input_list, axis=0)
        self.label = np.concatenate(self.label_list, axis=0)

        print(f'\ninput {self.data_type} dataset size: {np.shape(self.input)}')
        print(f'label {self.data_type} dataset size: {np.shape(self.label)}')

        # Calculate total number of sequences (reference approach: no NaN filtering)
        self.length = len(self.input) - self.window_size + 1
        print(f"Total {self.data_type} sequences: {self.length}")

        # Calculate mean and std using the entire dataset (reference uses np.mean, not nanmean)
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
