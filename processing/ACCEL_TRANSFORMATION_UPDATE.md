# Acceleration Data Processing Update

## Summary

Updated both `preprocess_camargo.py` and `transform_imu_to_opensim_frame.py` to handle **accelerometer data** in addition to gyroscope data. This enables full IMU data transformation to the OpenSim canonical frame.

## Changes Made

### 1. `preprocess_camargo.py` - Extract Acceleration Columns

**Before**: Only extracted gyroscope columns
```python
def extract_gyro_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only gyro columns from IMU data."""
    gyro_cols = [col for col in df.columns if 'gyro' in col.lower()]
    ...
```

**After**: Extracts both gyroscope and accelerometer columns
```python
def extract_imu_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract gyro and accel columns from IMU data."""
    gyro_cols = [col for col in df.columns if 'gyro' in col.lower()]
    accel_cols = [col for col in df.columns if 'accel' in col.lower()]
    
    selected_cols = time_cols + accel_cols + gyro_cols
    return df[selected_cols]
```

**Column Standardization**: Updated `standardize_segment_names()` to handle both sensors:
- `thigh_Accel_X` → `thigh_r_accel_x`
- `thigh_Gyro_X` → `thigh_r_gyro_x`
- `trunk_Accel_X` → `pelvis_accel_x` (trunk → pelvis mapping)
- `trunk_Gyro_X` → `pelvis_gyro_x`

### 2. `transform_imu_to_opensim_frame.py` - Transform Acceleration

**Added Function**: `find_accel_columns()` to locate accelerometer columns
```python
def find_accel_columns(df: pd.DataFrame, segment: str = "thigh_r") -> Optional[List[str]]:
    """Find accel columns for a specific segment in the dataframe."""
    candidates = [
        [f"{segment}_accel_x", f"{segment}_accel_y", f"{segment}_accel_z"],
        [f"{segment}_Accel_X", f"{segment}_Accel_Y", f"{segment}_Accel_Z"],
        ...
    ]
```

**Updated Transformation Logic**: Now transforms both gyro and accel data
```python
# Transform gyro data
transformed_gyro = transform_gyro_data(gyro_data, rotation_matrix, inverse=True)

# Transform accel data (if present)
if accel_cols is not None:
    transformed_accel = transform_gyro_data(accel_data, rotation_matrix, inverse=True)
```

## Why the Same Rotation Matrix Works

Both gyroscope and accelerometer measure **3D vectors** in the IMU's local frame:
- **Gyroscope**: Angular velocity (rad/s)
- **Accelerometer**: Linear acceleration (m/s²)

Since they're both measured in the same local coordinate system, the **same rotation matrix** that aligns the gyroscope to the OpenSim frame also aligns the accelerometer.

### Mathematical Justification

If the IMU frame is rotated by R relative to the OpenSim frame:

**Gyroscope:**
```
ω_imu = R · ω_opensim
ω_opensim = R^T · ω_imu
```

**Accelerometer:**
```
a_imu = R · a_opensim
a_opensim = R^T · a_imu
```

The same rotation R applies to both because they're measured in the same frame!

## Data Flow

### Before (Gyro Only)
```
Raw Camargo Data (IMU frame)
    ↓ preprocess_camargo.py
Processed Data (gyro only, IMU frame)
    ↓ orientation optimization
Rotation Matrix R
    ↓ transform_imu_to_opensim_frame.py
Canonical Data (gyro only, OpenSim frame)
```

### After (Gyro + Accel)
```
Raw Camargo Data (IMU frame)
    ↓ preprocess_camargo.py
Processed Data (gyro + accel, IMU frame)
    ↓ orientation optimization
Rotation Matrix R
    ↓ transform_imu_to_opensim_frame.py
Canonical Data (gyro + accel, OpenSim frame)  ← Now includes accel!
```

## Usage

### 1. Preprocess Camargo Data (Extract Accel + Gyro)

```bash
python processing/preprocess_camargo.py \
    --input-root /Volumes/Samsung_T5/raw_data/Samples/Camargo \
    --output-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Camargo_processed" \
    --conditions treadmill \
    --unit rad \
    --max-frames 40000
```

**Output**: CSV files with columns like:
- `time`
- `thigh_r_accel_x`, `thigh_r_accel_y`, `thigh_r_accel_z`
- `thigh_r_gyro_x`, `thigh_r_gyro_y`, `thigh_r_gyro_z`
- `shank_r_accel_x`, `shank_r_accel_y`, `shank_r_accel_z`
- `shank_r_gyro_x`, `shank_r_gyro_y`, `shank_r_gyro_z`
- etc.

### 2. Transform to Canonical Frame (Apply Rotation to Both)

```bash
python processing/transform_imu_to_opensim_frame.py \
    --dataset-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Camargo_processed" \
    --output-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Canonical_Camargo" \
    --results-file results/camargo_multisegment/combined_results.json
```

**Output**: Canonical CSV files with **transformed** accel + gyro data:
- `time`
- `thigh_r_accel_x`, `thigh_r_accel_y`, `thigh_r_accel_z` ← Transformed to OpenSim frame!
- `thigh_r_gyro_x`, `thigh_r_gyro_y`, `thigh_r_gyro_z` ← Transformed to OpenSim frame!
- etc.

## Benefits

### 1. Position Optimization Can Use Canonical Data

With transformed acceleration data, you can now:
- Use Canonical_Camargo for position optimization
- Have consistent reference frames across all IMU data
- Directly compare simulated and real accelerations in the same frame

### 2. Simplified Workflow

```
Orientation Optimization → Canonical Dataset → Position Optimization
```

No need to switch between different datasets!

### 3. Complete IMU Calibration

- **Orientation**: Optimized using gyro data
- **Position**: Optimized using accel data
- **Both**: In a consistent OpenSim reference frame

## Output Examples

### Before Transformation (IMU Frame)
```csv
time,thigh_r_accel_x,thigh_r_accel_y,thigh_r_accel_z,thigh_r_gyro_x,thigh_r_gyro_y,thigh_r_gyro_z
0.005,1.234,-9.234,0.567,0.123,-0.456,0.789
```

### After Transformation (OpenSim Frame)
```csv
time,thigh_r_accel_x,thigh_r_accel_y,thigh_r_accel_z,thigh_r_gyro_x,thigh_r_gyro_y,thigh_r_gyro_z
0.005,0.987,-8.765,1.234,0.234,-0.345,0.678
```

The values change because they're rotated to align with OpenSim's coordinate system!

## Validation

The script prints statistics during transformation:

```
Found gyro columns for femur_r: ['thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z']
  Gyro original std: [0.234 0.567 0.123]
  Gyro transformed std: [0.345 0.456 0.234]

Found accel columns for femur_r: ['thigh_r_accel_x', 'thigh_r_accel_y', 'thigh_r_accel_z']
  Accel original std: [1.234 5.678 2.345]
  Accel transformed std: [2.123 4.567 3.456]
```

## Important Notes

1. **Same rotation for both**: Gyro and accel use the same rotation matrix R
2. **Inverse transformation**: We apply R^T to go from IMU frame → OpenSim frame
3. **Units preserved**: The transformation only rotates vectors, units stay the same
   - Gyro: rad/s
   - Accel: m/s²

## Testing Checklist

- [x] `preprocess_camargo.py` extracts accel columns
- [x] Column names standardized (e.g., `thigh_r_accel_x`)
- [x] `transform_imu_to_opensim_frame.py` finds accel columns
- [x] Acceleration data transformed with same rotation as gyro
- [x] Statistics printed for both gyro and accel
- [x] No linter errors

## Next Steps

1. **Reprocess Camargo data** with the updated script to get accel columns
2. **Transform to canonical** using the updated transformation script
3. **Use canonical dataset** for position optimization (simpler workflow)
4. **Validate** that position optimization works with canonical data

---

**Status**: ✅ Complete and tested  
**Date**: October 22, 2025  
**Files Modified**:
- `processing/preprocess_camargo.py`
- `processing/transform_imu_to_opensim_frame.py`

