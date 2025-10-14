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
        self.use_subject_info = hyperparam_config.get("use_subject_info", False)
        
        # Detect dataset type for sampling rate adjustment
        self.dataset_type = self._detect_dataset_type()
        print(f"Detected dataset type: {self.dataset_type}")
            
        # Placeholder for data
        self.train_data = None
        self.test_data = None

        # Initialize mean and std attributes
        self.input_mean = None
        self.input_std = None
        self.label_mean = None
        self.label_std = None
        self.param_mean = None
        self.param_std = None

        # Load subject information if requested
        self.subject_info = None
        if self.use_subject_info:
            subject_info_path = os.path.join(self.data_root, 'SubjectInfo.csv')
            if os.path.exists(subject_info_path):
                try:
                    self.subject_info = pd.read_csv(subject_info_path)
                except Exception as exc:
                    raise RuntimeError(f"Failed to read subject information from {subject_info_path}: {exc}")
            else:
                raise FileNotFoundError(
                    f"Subject information requested but SubjectInfo.csv not found at {subject_info_path}"
                )

        # (For transfer learning) Load saved mean and std for normalization
        if self.pretrained_model_path is not None:
            self.input_mean = np.load(os.path.join(self.pretrained_model_path, 'input_mean.npy'))
            self.input_std = np.load(os.path.join(self.pretrained_model_path, 'input_std.npy'))
            self.label_mean = np.load(os.path.join(self.pretrained_model_path, 'label_mean.npy'))
            self.label_std = np.load(os.path.join(self.pretrained_model_path, 'label_std.npy'))

            param_mean_path = os.path.join(self.pretrained_model_path, 'param_mean.npy')
            param_std_path = os.path.join(self.pretrained_model_path, 'param_std.npy')
            if os.path.exists(param_mean_path) and os.path.exists(param_std_path):
                self.param_mean = np.load(param_mean_path)
                self.param_std = np.load(param_std_path)
    
    def _detect_dataset_type(self) -> str:
        """Detect dataset type based on data_root path to determine sampling rate."""
        if 'Canonical_Camargo' in self.data_root:
            return 'camargo'  # Higher sampling rate, needs downsampling
        elif 'Canonical_MeMo' in self.data_root:
            return 'memo'     # Standard sampling rate
        else:
            return 'unknown'  # Default to no downsampling
        
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
            label_filter_hz=self.label_filter_hz,
            dataset_type=self.dataset_type,
            subject_info_df=self.subject_info if self.use_subject_info else None
        )
        
        # Retrieve mean and std from training data
        self.input_mean = self.train_data.input_mean
        self.input_std = self.train_data.input_std
        self.label_mean = self.train_data.label_mean
        self.label_std = self.train_data.label_std
        if self.use_subject_info:
            self.param_mean = self.train_data.param_mean
            self.param_std = self.train_data.param_std
        
        # Load test data using training mean and std
        print("\n...Loading test data...\n")        
        self.test_data = LoadData(
            root=self.data_root,
            partitions=test_data_partition,
            conditions=train_data_condition,  # Use the same conditions as training data
            window_size=self.window_size,
            data_type="test_data",
            dataset_proportion=self.dataset_proportion,
            input_mean=self.input_mean,
            input_std=self.input_std,
            label_mean=self.label_mean,
            label_std=self.label_std,
            normalize=self.normalize,
            imu_segments=self.imu_segments,
            augment=False,
            label_filter_hz=self.label_filter_hz,
            dataset_type=self.dataset_type,
            subject_info_df=self.subject_info if self.use_subject_info else None,
            param_mean=self.param_mean,
            param_std=self.param_std,
        )
        # Replace the mean and std of test data with that of training data
        self.test_data.input_mean = self.input_mean
        self.test_data.input_std = self.input_std
        self.test_data.label_mean = self.label_mean
        self.test_data.label_std = self.label_std
        if self.use_subject_info:
            self.test_data.param_mean = self.param_mean
            self.test_data.param_std = self.param_std
        
    def save_mean_std(self, save_dir: str):
        """Save mean and std to .npy files."""
        np.save(os.path.join(save_dir, 'input_mean.npy'), self.input_mean)
        np.save(os.path.join(save_dir, 'input_std.npy'), self.input_std)
        np.save(os.path.join(save_dir, 'label_mean.npy'), self.label_mean)
        np.save(os.path.join(save_dir, 'label_std.npy'), self.label_std)
        if self.param_mean is not None and self.param_std is not None:
            np.save(os.path.join(save_dir, 'param_mean.npy'), self.param_mean)
            np.save(os.path.join(save_dir, 'param_std.npy'), self.param_std)
     
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

    def load_test_data_only(self, test_data_partition: List[str], conditions: List[str]) -> None:
        """Load only the test split using stored normalization statistics."""
        if not test_data_partition:
            raise ValueError("At least one test partition must be provided")

        if self.input_mean is None or self.input_std is None or self.label_mean is None or self.label_std is None:
            raise RuntimeError(
                "Pretrained normalization statistics were not found. "
                "Instantiate DataHandler with pretrained_model_path pointing to a trained model directory."
            )

        print("\n...Loading test data...\n")
        self.test_data = LoadData(
            root=self.data_root,
            partitions=test_data_partition,
            conditions=conditions,
            window_size=self.window_size,
            data_type="test_data",
            dataset_proportion=self.dataset_proportion,
            normalize=self.normalize,
            imu_segments=self.imu_segments,
            augment=False,
            label_filter_hz=self.label_filter_hz,
            dataset_type=self.dataset_type,
            subject_info_df=self.subject_info if self.use_subject_info else None,
            input_mean=self.input_mean,
            input_std=self.input_std,
            label_mean=self.label_mean,
            label_std=self.label_std,
            param_mean=self.param_mean,
            param_std=self.param_std,
        )

        if self.use_subject_info:
            # Ensure downstream code can access parameter statistics
            self.test_data.param_mean = self.param_mean
            self.test_data.param_std = self.param_std


class LoadData(Dataset):
    """Dataset class to load data from Canonical format."""

    def __init__(
        self,
        root: str,
        partitions: List[str],
        conditions: List[str],
        window_size: int,
        data_type: str,
        dataset_proportion: Optional[float] = None,
        input_mean: Optional[np.ndarray] = None,
        input_std: Optional[np.ndarray] = None,
        label_mean: Optional[np.ndarray] = None,
        label_std: Optional[np.ndarray] = None,
        normalize: bool = True,
        imu_segments: Optional[List[str]] = None,
        augment: bool = False,
        label_filter_hz: float = 6.0,
        dataset_type: str = 'unknown',
        subject_info_df: Optional[pd.DataFrame] = None,
        param_mean: Optional[np.ndarray] = None,
        param_std: Optional[np.ndarray] = None,
    ):
        self.window_size = window_size
        self.data_type = data_type
        self.dataset_proportion = dataset_proportion
        self.normalize = normalize
        self.imu_segments = imu_segments if imu_segments is not None else ["pelvis", "femur"]
        self.augment = augment and data_type.startswith("train")
        self.label_filter_hz = float(label_filter_hz)
        self.dataset_type = dataset_type
        self.subject_info_df = subject_info_df.copy() if subject_info_df is not None else None
        self.use_subject_info = self.subject_info_df is not None

        self.input_list: List[np.ndarray] = []
        self.label_list: List[np.ndarray] = []
        self.param_list: List[np.ndarray] = []
        self.subject_data_length: List[int] = []

        self.input: Optional[np.ndarray] = None
        self.label: Optional[np.ndarray] = None
        self.subject_params: Optional[np.ndarray] = None

        self.input_mean: Optional[np.ndarray] = None
        self.input_std: Optional[np.ndarray] = None
        self.label_mean: Optional[np.ndarray] = None
        self.label_std: Optional[np.ndarray] = None
        self.param_mean: Optional[np.ndarray] = None
        self.param_std: Optional[np.ndarray] = None

        self._prepare_subject_info()
        self._collect_samples(root, partitions, conditions)
        self._finalize_arrays()
        self._downsample_if_needed()
        self._compute_statistics()
        self._apply_overrides(input_mean, input_std, label_mean, label_std, param_mean, param_std)

    def _prepare_subject_info(self) -> None:
        if not self.use_subject_info:
            self.subject_info_lookup = {}
            return

        columns_lower = {col.lower(): col for col in self.subject_info_df.columns}

        subject_col = None
        for candidate in ["subject", "id", "participant"]:
            if candidate in columns_lower:
                subject_col = columns_lower[candidate]
                break
        if subject_col is None:
            raise ValueError("SubjectInfo.csv must contain a 'Subject' column identifying participants")

        mass_col = None
        for candidate in ["mass", "body_mass", "weight"]:
            if candidate in columns_lower:
                mass_col = columns_lower[candidate]
                break
        if mass_col is None:
            raise ValueError("SubjectInfo.csv must contain a 'Mass' column (mass/body_mass/weight)")

        height_col = None
        for candidate in ["height", "stature"]:
            if candidate in columns_lower:
                height_col = columns_lower[candidate]
                break
        if height_col is None:
            raise ValueError("SubjectInfo.csv must contain a 'Height' column (height/stature)")

        self.subject_info_df = self.subject_info_df.rename(
            columns={subject_col: "Subject", mass_col: "Mass", height_col: "Height"}
        )
        self.subject_info_df["Subject"] = self.subject_info_df["Subject"].astype(str)

        self.subject_info_lookup = {
            row["Subject"].strip(): (float(row["Mass"]), float(row["Height"]))
            for _, row in self.subject_info_df.iterrows()
        }

    def _lookup_subject_params(self, subject_id: str) -> Tuple[float, float]:
        if not self.use_subject_info:
            return 0.0, 0.0

        candidates = {
            subject_id,
            subject_id.strip(),
            subject_id.upper(),
            subject_id.lower(),
            subject_id.strip().upper(),
            subject_id.strip().lower(),
        }

        stripped = subject_id.lstrip('0')
        if stripped:
            candidates.update({stripped, stripped.upper(), stripped.lower()})

        for candidate in candidates:
            if candidate in self.subject_info_lookup:
                return self.subject_info_lookup[candidate]

        print(f"⚠️ Subject parameters not found for {subject_id}. Using NaN placeholders.")
        return float('nan'), float('nan')

    def _collect_samples(self, root: str, partitions: List[str], conditions: List[str]) -> None:
        for subject_num, subject in enumerate(partitions):
            self.subject_data_length.append(0)
            subject_path = os.path.join(root, subject)
            if not os.path.isdir(subject_path):
                continue

            if self.use_subject_info:
                subject_mass, subject_height = self._lookup_subject_params(subject)
                subject_param_vec = np.array([subject_mass, subject_height], dtype=np.float32)
            else:
                subject_param_vec = None

            for condition in conditions:
                condition_path = os.path.join(subject_path, condition)
                if not os.path.isdir(condition_path):
                    continue

                for trial in os.listdir(condition_path):
                    trial_path = os.path.join(condition_path, trial)
                    if not os.path.isdir(trial_path):
                        continue

                    input_file_dir = os.path.join(trial_path, 'Input')
                    label_file_dir = os.path.join(trial_path, 'Label')
                    if not os.path.exists(input_file_dir) or not os.path.exists(label_file_dir):
                        continue

                    input_file_names = sorted(os.listdir(input_file_dir))
                    label_file_names = sorted(os.listdir(label_file_dir))
                    if not label_file_names:
                        continue

                    input_buffer_R = None
                    input_buffer_L = None

                    for name in input_file_names:
                        if 'imu' not in name.lower():
                            continue
                        csv_path = os.path.join(input_file_dir, name)
                        try:
                            df = pd.read_csv(csv_path, sep=None, engine='python', on_bad_lines='skip')
                        except Exception:
                            df = pd.read_csv(csv_path, sep=',', on_bad_lines='skip')

                        gyro_cols = [col for col in df.columns if 'gyro' in col.lower()]
                        right_cols: List[str] = []
                        left_cols: List[str] = []

                        if len(self.imu_segments) == 1 and self.imu_segments[0].lower() in ['femur', 'thigh']:
                            thigh_r_gyro = [col for col in gyro_cols if 'thigh_r' in col.lower() or 'femur_r' in col.lower()]
                            thigh_l_gyro = [col for col in gyro_cols if 'thigh_l' in col.lower() or 'femur_l' in col.lower()]
                            if not thigh_r_gyro or len(thigh_r_gyro) < 3:
                                raise ValueError(
                                    f"Required IMU segment 'femur/thigh' not found in {csv_path}\n"
                                    f"Available gyro columns: {gyro_cols}\n"
                                    f"Make sure your data contains 'thigh_r' or 'femur_r' gyro data."
                                )
                            right_cols = thigh_r_gyro[:3]
                            if thigh_l_gyro and len(thigh_l_gyro) >= 3:
                                left_cols = thigh_l_gyro[:3]
                        elif len(self.imu_segments) == 2:
                            seg1 = self.imu_segments[0].lower()
                            seg2 = self.imu_segments[1].lower()
                            pelvis_gyro = [col for col in gyro_cols if 'pelvis' in col.lower()]
                            thigh_r_gyro = [col for col in gyro_cols if 'thigh_r' in col.lower() or 'femur_r' in col.lower()]
                            thigh_l_gyro = [col for col in gyro_cols if 'thigh_l' in col.lower() or 'femur_l' in col.lower()]
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
                                    "Supported configurations:\n"
                                    "  - ['femur'] or ['thigh'] for single IMU\n"
                                    "  - ['pelvis', 'femur'] or ['pelvis', 'thigh'] for dual IMU"
                                )
                        else:
                            raise ValueError(
                                f"Invalid number of IMU segments: {len(self.imu_segments)}\n"
                                "Supported: 1 segment ['femur'] or 2 segments ['pelvis', 'femur']"
                            )

                        if not right_cols:
                            raise ValueError(
                                f"Failed to extract right side IMU data from {csv_path}\n"
                                f"Requested segments: {self.imu_segments}\n"
                                f"Available gyro columns: {gyro_cols}"
                            )

                        input_df_R = df[right_cols].values
                        input_df_L = df[left_cols].values if left_cols else None

                        if input_buffer_R is None:
                            input_buffer_R = input_df_R
                        else:
                            input_buffer_R = np.hstack((input_buffer_R, input_df_R))

                        if input_df_L is not None:
                            if input_buffer_L is None:
                                input_buffer_L = input_df_L
                            else:
                                input_buffer_L = np.hstack((input_buffer_L, input_df_L))

                    if input_buffer_R is None:
                        continue

                    label_file_path = os.path.join(label_file_dir, label_file_names[0])
                    try:
                        label_df = pd.read_csv(label_file_path, sep=None, engine='python', on_bad_lines='skip')
                    except Exception:
                        label_df = pd.read_csv(label_file_path, sep=',', on_bad_lines='skip')

                    hip_flexion_r_col = [col for col in label_df.columns if 'hip_flexion_r_moment' in col.lower()]
                    hip_flexion_l_col = [col for col in label_df.columns if 'hip_flexion_l_moment' in col.lower()]

                    label_buffer_R = None
                    label_buffer_L = None

                    if hip_flexion_r_col:
                        label_buffer_R = label_df[hip_flexion_r_col[0]].values.reshape(-1, 1)
                        if self.dataset_type == 'camargo':
                            label_buffer_R = -label_buffer_R
                    if hip_flexion_l_col:
                        label_buffer_L = label_df[hip_flexion_l_col[0]].values.reshape(-1, 1)

                    if label_buffer_R is None and label_buffer_L is None:
                        hip_moment_cols = [col for col in label_df.columns if 'hip' in col.lower() and 'moment' in col.lower()]
                        if hip_moment_cols:
                            hip_r_col = [col for col in hip_moment_cols if 'r' in col.lower() or 'right' in col.lower()]
                            hip_l_col = [col for col in hip_moment_cols if 'l' in col.lower() or 'left' in col.lower()]
                            if hip_r_col:
                                label_buffer_R = label_df[hip_r_col[0]].values.reshape(-1, 1)
                                if self.dataset_type == 'camargo':
                                    label_buffer_R = -label_buffer_R
                            if hip_l_col:
                                label_buffer_L = label_df[hip_l_col[0]].values.reshape(-1, 1)
                        if label_buffer_R is None and label_buffer_L is None:
                            continue

                    label_buffer_R = self._butter_lowpass_zero_phase(label_buffer_R)
                    label_buffer_L = self._butter_lowpass_zero_phase(label_buffer_L)

                    if input_buffer_R is not None and label_buffer_R is not None:
                        min_len_R = min(input_buffer_R.shape[0], label_buffer_R.shape[0])
                        input_buffer_R = input_buffer_R[:min_len_R]
                        label_buffer_R = label_buffer_R[:min_len_R]
                        valid_mask_R = (~np.isnan(input_buffer_R).any(axis=1)) & (~np.isnan(label_buffer_R).any(axis=1))
                        input_buffer_R_clean = input_buffer_R[valid_mask_R]
                        label_buffer_R_clean = label_buffer_R[valid_mask_R]
                    else:
                        input_buffer_R_clean = None
                        label_buffer_R_clean = None

                    if input_buffer_L is not None and label_buffer_L is not None:
                        min_len_L = min(input_buffer_L.shape[0], label_buffer_L.shape[0])
                        input_buffer_L = input_buffer_L[:min_len_L]
                        label_buffer_L = label_buffer_L[:min_len_L]
                        valid_mask_L = (~np.isnan(input_buffer_L).any(axis=1)) & (~np.isnan(label_buffer_L).any(axis=1))
                        input_buffer_L_clean = input_buffer_L[valid_mask_L]
                        label_buffer_L_clean = label_buffer_L[valid_mask_L]
                    else:
                        input_buffer_L_clean = None
                        label_buffer_L_clean = None

                    input_buffer_R_clean = self._apply_augmentation(input_buffer_R_clean)
                    if input_buffer_L_clean is not None:
                        input_buffer_L_clean = self._apply_augmentation(input_buffer_L_clean)

                    if input_buffer_R_clean is not None and len(input_buffer_R_clean) > 0:
                        if input_buffer_L_clean is not None and len(input_buffer_L_clean) > 0:
                            if np.random.randint(0, 2):
                                input_buffer = np.vstack((input_buffer_R_clean, input_buffer_L_clean))
                                label_buffer = np.vstack((label_buffer_R_clean, label_buffer_L_clean))
                            else:
                                input_buffer = np.vstack((input_buffer_L_clean, input_buffer_R_clean))
                                label_buffer = np.vstack((label_buffer_L_clean, label_buffer_R_clean))
                        else:
                            input_buffer = input_buffer_R_clean
                            label_buffer = label_buffer_R_clean
                    elif input_buffer_L_clean is not None and len(input_buffer_L_clean) > 0:
                        input_buffer = input_buffer_L_clean
                        label_buffer = label_buffer_L_clean
                    else:
                        continue

                    self.input_list.append(input_buffer)
                    self.label_list.append(label_buffer)
                    if self.use_subject_info and subject_param_vec is not None:
                        param_buffer = np.repeat(subject_param_vec[np.newaxis, :], input_buffer.shape[0], axis=0)
                        self.param_list.append(param_buffer)
                    self.subject_data_length[subject_num] += input_buffer.shape[0]

            print(f"{subject} data length: {self.subject_data_length[subject_num]}")

        if not self.input_list or not self.label_list:
            raise ValueError("No valid data found in the specified partitions and conditions")

    def _butter_lowpass_zero_phase(
        self,
        data: Optional[np.ndarray],
        cutoff_hz: Optional[float] = None,
        fs_hz: float = 100.0,
        order: int = 4,
    ) -> Optional[np.ndarray]:
        if data is None:
            return None
        if data.size == 0:
            return data
        nyq = 0.5 * fs_hz
        cutoff = self.label_filter_hz if cutoff_hz is None else cutoff_hz
        wn = float(cutoff) / nyq
        b, a = signal.butter(order, wn, btype='low', analog=False)
        try:
            filtered = signal.filtfilt(
                b,
                a,
                data.squeeze(),
                axis=0,
                method='pad',
                padlen=min(3 * max(len(a), len(b)), max(0, len(data) - 1)),
            )
        except ValueError:
            y = signal.lfilter(b, a, data.squeeze(), axis=0)
            filtered = signal.lfilter(b, a, y[::-1], axis=0)[::-1]
        return filtered.reshape(-1, 1)

    def _apply_augmentation(self, arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if not self.augment or arr is None or len(arr) == 0:
            return arr
        data = arr.copy()
        num_channels = data.shape[1]
        if num_channels % 3 == 0:
            num_sensors = num_channels // 3
            for sensor_idx in range(num_sensors):
                sl = slice(3 * sensor_idx, 3 * sensor_idx + 3)
                gx = np.deg2rad(np.random.uniform(-10, 10))
                gy = np.deg2rad(np.random.uniform(-10, 10))
                gz = np.deg2rad(np.random.uniform(-10, 10))
                Rx = np.array([[1, 0, 0], [0, np.cos(gx), -np.sin(gx)], [0, np.sin(gx), np.cos(gx)]])
                Ry = np.array([[np.cos(gy), 0, np.sin(gy)], [0, 1, 0], [-np.sin(gy), 0, np.cos(gy)]])
                Rz = np.array([[np.cos(gz), -np.sin(gz), 0], [np.sin(gz), np.cos(gz), 0], [0, 0, 1]])
                R = Rz @ Ry @ Rx
                data[:, sl] = data[:, sl] @ R.T
        noise = np.random.normal(scale=0.01, size=data.shape)
        return data + noise

    def _finalize_arrays(self) -> None:
        self.input = np.concatenate(self.input_list, axis=0)
        self.label = np.concatenate(self.label_list, axis=0)
        if self.use_subject_info:
            if not self.param_list:
                raise ValueError("Subject information was requested but no parameters were collected.")
            self.subject_params = np.concatenate(self.param_list, axis=0)
        else:
            self.subject_params = None
        self.length = len(self.input) - self.window_size + 1
        print(f"\ninput {self.data_type} dataset size: {np.shape(self.input)}")
        print(f"label {self.data_type} dataset size: {np.shape(self.label)}")
        print(f"Total {self.data_type} sequences: {self.length}")

    def _downsample_if_needed(self) -> None:
        if self.dataset_type != 'camargo':
            return
        print("Applying downsampling (::2) for Camargo dataset...")
        original_input_size = self.input.shape[0]
        original_label_size = self.label.shape[0]
        self.input = self.input[::2]
        self.label = self.label[::2]
        if self.subject_params is not None:
            self.subject_params = self.subject_params[::2]
        for idx in range(len(self.subject_data_length)):
            self.subject_data_length[idx] = self.subject_data_length[idx] // 2
        print(f"Downsampled input: {original_input_size} -> {self.input.shape[0]} samples")
        print(f"Downsampled label: {original_label_size} -> {self.label.shape[0]} samples")
        self.length = len(self.input) - self.window_size + 1
        print(f"Total {self.data_type} sequences after downsampling: {self.length}")

    def _compute_statistics(self) -> None:
        self.input_mean = np.mean(self.input, axis=0)
        self.input_std = np.std(self.input, axis=0) + 1e-8
        self.label_mean = np.mean(self.label, axis=0)
        self.label_std = np.std(self.label, axis=0) + 1e-8
        if self.subject_params is not None:
            self.param_mean = np.nanmean(self.subject_params, axis=0)
            self.param_std = np.nanstd(self.subject_params, axis=0) + 1e-8
            nan_mask = np.isnan(self.subject_params)
            if np.any(nan_mask):
                nan_replacements = np.where(np.isnan(self.param_mean), 0.0, self.param_mean)
                rows, cols = np.where(nan_mask)
                self.subject_params[rows, cols] = nan_replacements[cols]
        else:
            self.param_mean = None
            self.param_std = None

    def _apply_overrides(
        self,
        input_mean: Optional[np.ndarray],
        input_std: Optional[np.ndarray],
        label_mean: Optional[np.ndarray],
        label_std: Optional[np.ndarray],
        param_mean: Optional[np.ndarray],
        param_std: Optional[np.ndarray],
    ) -> None:
        if input_mean is not None:
            self.input_mean = input_mean
        if input_std is not None:
            self.input_std = input_std
        if label_mean is not None:
            self.label_mean = label_mean
        if label_std is not None:
            self.label_std = label_std
        if param_mean is not None:
            self.param_mean = param_mean
        if param_std is not None:
            self.param_std = param_std

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, ind: int):
        windows_input = self.input[ind: ind + self.window_size]
        if self.normalize:
            windows_input = (windows_input - self.input_mean) / self.input_std
        window_input = torch.FloatTensor(windows_input).T
        target_label = self.label[ind + self.window_size - 1]
        if self.normalize:
            target_label = (target_label - self.label_mean) / self.label_std
        window_label = torch.FloatTensor(target_label)
        if self.use_subject_info and self.subject_params is not None:
            params = self.subject_params[ind + self.window_size - 1]
            if self.normalize and self.param_mean is not None and self.param_std is not None:
                params = (params - self.param_mean) / self.param_std
            param_tensor = torch.FloatTensor(params)
            return window_input, window_label, param_tensor
        return window_input, window_label
