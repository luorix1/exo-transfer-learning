# MeMo Dataset Processing Guide

## Overview

The `process_memo.py` script processes the MeMo_processed dataset into the standardized Canonical format with proper coordinate frame transformation.

## Key Features

1. **Automatic File Renaming**: Converts label files from `{subject}_{condition}_{trial}.csv` to `joint_moment.csv`
2. **Column Standardization**: Converts all column names to lowercase with underscores
3. **Coordinate Frame Transformation**: Transforms IMU gyroscope data from MeMo frame to OpenSim canonical frame
4. **Unit Conversion**: Converts joint moments from N·mm/kg to N·m/kg (divides by 1000)

## Coordinate Systems

### MeMo IMU Frame
The MeMo dataset uses the following IMU coordinate convention:
- **X-axis**: Up (vertical, pointing upward)
- **Y-axis**: Left (mediolateral, pointing left)
- **Z-axis**: Back (anterior-posterior, pointing backward)

### OpenSim Canonical Frame
OpenSim uses a different standard coordinate system:
- **X-axis**: Forward (anterior-posterior, pointing forward)
- **Y-axis**: Up (vertical, pointing upward)
- **Z-axis**: Right (mediolateral, pointing right)

### Transformation Matrix

The transformation from MeMo to Canonical is:
```
canonical_x (forward) = -memo_z  (flip back → forward)
canonical_y (up)      =  memo_x  (up stays up)
canonical_z (right)   = -memo_y  (flip left → right)
```

In matrix form:
```
[canonical_x]   [ 0  0 -1] [memo_x]
[canonical_y] = [ 1  0  0] [memo_y]
[canonical_z]   [ 0 -1  0] [memo_z]
```

## Usage Examples

### Process All Subjects and Conditions

```bash
python processing/process_memo.py \
  --input-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/MeMo_processed" \
  --output-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Canonical_MeMo"
```

### Process Specific Subjects

```bash
python processing/process_memo.py \
  --input-root "/path/to/MeMo_processed" \
  --output-root "/path/to/Canonical_MeMo" \
  --subjects AB01_Jimin,AB02_Rajiv,AB03_Amy
```

### Process Specific Conditions

```bash
python processing/process_memo.py \
  --input-root "/path/to/MeMo_processed" \
  --output-root "/path/to/Canonical_MeMo" \
  --conditions 0mps,1p0mps,1p4mps
```

### Process Without Frame Transformation

If you want to keep the original MeMo frame (not recommended for use with OpenSim-trained models):

```bash
python processing/process_memo.py \
  --input-root "/path/to/MeMo_processed" \
  --output-root "/path/to/Canonical_MeMo_NoTransform" \
  --no-transform
```

### Process Subset for Testing

```bash
python processing/process_memo.py \
  --input-root "/path/to/MeMo_processed" \
  --output-root "./test_canonical_memo" \
  --subjects AB01_Jimin \
  --conditions 0mps
```

## Input Structure

The script expects the MeMo_processed structure:
```
MeMo_processed/
├── AB01_Jimin/
│   ├── 0mps/
│   │   ├── trial_1/
│   │   │   ├── Input/
│   │   │   │   └── imu_data.csv
│   │   │   └── Label/
│   │   │       └── AB01_Jimin_0mps_1.csv
│   │   ├── trial_2/
│   │   └── trial_3/
│   ├── 1p0mps/
│   └── ...
├── AB02_Rajiv/
└── ...
```

## Output Structure

The script produces the standardized Canonical format:
```
Canonical_MeMo/
├── AB01_Jimin/
│   ├── 0mps/
│   │   ├── trial_1/
│   │   │   ├── Input/
│   │   │   │   └── imu_data.csv          # Transformed & standardized
│   │   │   └── Label/
│   │   │       └── joint_moment.csv      # Renamed & standardized
│   │   ├── trial_2/
│   │   └── trial_3/
│   ├── 1p0mps/
│   └── ...
├── AB02_Rajiv/
└── ...
```

## Data Transformations

### IMU Data (Input/imu_data.csv)

**Before (Original MeMo):**
```csv
time,Pelvis_Gyr_X,Pelvis_Gyr_Y,Pelvis_Gyr_Z,Thigh_R_Gyr_X,Thigh_R_Gyr_Y,Thigh_R_Gyr_Z
0.0,-1.74,2.02,-1.18,0.37,-0.60,0.52
```

**After (Canonical):**
```csv
time,pelvis_gyr_x,pelvis_gyr_y,pelvis_gyr_z,thigh_r_gyr_x,thigh_r_gyr_y,thigh_r_gyr_z
0.0,1.18,-1.74,-2.02,-0.52,0.37,0.60
```

**Changes:**
1. Column names: `Pelvis_Gyr_X` → `pelvis_gyr_x` (lowercase, underscores)
2. Gyroscope values: Transformed from MeMo frame to canonical frame
3. Accelerometer values: Names standardized but NOT transformed (acceleration is in global frame)

### Label Data (Label/joint_moment.csv)

**Before (Original MeMo - N·mm/kg):**
```csv
Frame,LHipMoment_X,LHipMoment_Y,RHipMoment_X,RHipMoment_Y
2,1103.64,796.98,-99.14,-510.46
```

**After (Canonical - N·m/kg):**
```csv
frame,lhipmoment_x,lhipmoment_y,rhipmoment_x,rhipmoment_y
2,1.10364,0.79698,-0.09914,-0.51046
```

**Changes:**
1. Column names standardized to lowercase with underscores
2. **Moment values divided by 1000** to convert from N·mm/kg to N·m/kg
3. Joint moment values NOT rotated (they're already in body segment frames, not sensor frames)

## Verification

After processing, you can verify the transformation worked correctly:

```bash
# Check original data
head -2 "/path/to/MeMo_processed/AB01_Jimin/0mps/trial_1/Input/imu_data.csv"

# Check transformed data
head -2 "/path/to/Canonical_MeMo/AB01_Jimin/0mps/trial_1/Input/imu_data.csv"

# Verify gyroscope transformation:
# If original pelvis gyro was [x, y, z] = [-1.74, 2.02, -1.18]
# Canonical should be [x, y, z] = [1.18, -1.74, -2.02]
```

## Using with Training Pipeline

After processing, the Canonical_MeMo dataset can be used directly with the training pipeline:

```bash
# Train on MeMo data
python src/train.py \
  --data_root "/path/to/Canonical_MeMo" \
  --save_dir "./checkpoints_memo" \
  --train_subjects AB01_Jimin,AB02_Rajiv,AB03_Amy \
  --test_subjects AB04_Changseob \
  --conditions 0mps,1p0mps \
  --imu_segments pelvis thigh_r \
  --epochs 30
```

## Common Issues

### Issue: "No trials found matching criteria"
**Solution**: Check that subjects and conditions match the exact folder names in MeMo_processed (case-sensitive, including underscores).

### Issue: "Incomplete gyroscope axes"
**Solution**: This warning appears if the IMU data doesn't have all three axes (X, Y, Z). Check the input CSV has complete gyroscope columns.

### Issue: Frame transformation seems wrong
**Solution**: Verify you're using the correct input frame convention. MeMo uses x=up, y=left, z=back. If your data uses a different convention, the transformation matrix needs adjustment.

## Technical Notes

- Only gyroscope data is transformed (angular velocity is in the sensor frame)
- Accelerometer data column names are standardized but NOT transformed (acceleration is typically in global/world frame)
- Joint moment data is NOT transformed (it's already in body segment frames)
- Time columns are preserved as-is
- All transformations are applied element-wise per sample

## Performance

Processing time depends on dataset size:
- Single subject, single condition (~3 trials): < 5 seconds
- Full MeMo dataset (~14 subjects, multiple conditions): ~2-5 minutes

Memory usage is minimal as files are processed one trial at a time.

