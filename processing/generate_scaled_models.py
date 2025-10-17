#!/usr/bin/env python3
"""
Generate scaled OpenSim models for each subject using a given OpenSim model
and subject information from SubjectInfo.csv.

This script creates personalized OpenSim models by scaling the generic model based on
each subject's height and weight measurements.

Usage:
    python generate_models.py <dataset_directory>

Example:
    python generate_scaled_models.py "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Keaton" \
      --model-file "/Users/luorix/Documents/OpenSim/Resources/Models/Gait2392_SimBody/gait2392_simbody.osim"
"""

import os
import sys
import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import shutil
from typing import Dict, Tuple, Optional
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class OpenSimModelGenerator:
    """
    Generates scaled OpenSim models for subjects based on their anthropometric data.
    """
    
    def __init__(self, dataset_dir: str, model_file: str = "/Users/luorix/Documents/OpenSim/Resources/Models/Gait2392_SimBody/gait2392_simbody.osim"):
        """
        Initialize the model generator.
        
        Args:
            dataset_dir: Path to the dataset directory containing SubjectInfo.csv
            model_file: Path to the generic OpenSim model file
        """
        self.dataset_dir = Path(dataset_dir)
        self.model_file = Path(model_file)
        self.subject_info_file = self.dataset_dir / "SubjectInfo.csv"
        
        # Validate inputs
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        
        if not self.model_file.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_file}")
        
        if not self.subject_info_file.exists():
            logger.warning(f"SubjectInfo.csv not found: {self.subject_info_file} — proceeding with default scaling (no SubjectInfo).")
        
        logger.info(f"Initialized model generator:")
        logger.info(f"  Dataset directory: {self.dataset_dir}")
        logger.info(f"  Model file: {self.model_file}")
        logger.info(f"  Subject info file: {self.subject_info_file}")
    
    def load_subject_info(self) -> pd.DataFrame:
        """
        Load subject information from SubjectInfo.csv.
        
        Returns:
            DataFrame with subject information
        """
        if not self.subject_info_file.exists():
            logger.info("No SubjectInfo.csv present; using empty subject info.")
            return pd.DataFrame(columns=["Subject","Age","Gender","Height","Weight"])
        try:
            df = pd.read_csv(self.subject_info_file)
            # Normalize columns (case-insensitive, common aliases)
            colmap = {c.lower(): c for c in df.columns}
            def pick(keys):
                for k in keys:
                    if k in colmap: return colmap[k]
                return None
            subj_col = pick(["subject","id","subject_id"])
            height_col = pick(["height","stature","height_m","height_cm"]) 
            weight_col = pick(["weight","mass","mass_kg","weight_kg"]) 
            rename = {}
            if subj_col and subj_col != 'Subject': rename[subj_col] = 'Subject'
            if height_col and height_col != 'Height': rename[height_col] = 'Height'
            if weight_col and weight_col != 'Weight': rename[weight_col] = 'Weight'
            if rename:
                df = df.rename(columns=rename)
            # Ensure required columns exist
            for req in ["Subject","Height","Weight"]:
                if req not in df.columns:
                    raise ValueError(f"SubjectInfo.csv missing required column: {req}")
            # Unit fix: if Height appears to be in centimeters (>3.0 typical meters), convert to meters
            if (df['Height'].dropna().astype(float) > 3.0).any():
                df['Height'] = df['Height'].astype(float) / 100.0
            logger.info(f"Loaded subject information for {len(df)} subjects")
            logger.debug(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Error loading subject info: {e}")
            return pd.DataFrame(columns=["Subject","Age","Gender","Height","Weight"])
    
    def get_subject_data(self, subject_id: str, subject_df: pd.DataFrame) -> Optional[Dict]:
        """
        Get anthropometric data for a specific subject.
        
        Args:
            subject_id: Subject ID (e.g., 'AB01')
            subject_df: DataFrame containing subject information
            
        Returns:
            Dictionary with subject data or None if not found
        """
        if subject_df is None or subject_df.empty or 'Subject' not in subject_df.columns:
            # Defaults if no subject info
            logger.warning(f"No SubjectInfo available; using defaults for {subject_id} (Height=1.75, Weight=75.0)")
            return {"Subject": subject_id, "Height": 1.75, "Weight": 75.0}
        subject_row = subject_df[subject_df['Subject'] == subject_id]
        
        if subject_row.empty:
            logger.warning(f"Subject {subject_id} not found in SubjectInfo.csv; using defaults (Height=1.75, Weight=75.0)")
            return {"Subject": subject_id, "Height": 1.75, "Weight": 75.0}
        
        subject_data = subject_row.iloc[0].to_dict()
        logger.debug(f"Subject {subject_id} data: {subject_data}")
        return subject_data
    
    def calculate_scaling_factors(self, subject_data: Dict) -> Dict[str, float]:
        """
        Calculate scaling factors based on subject's height and weight.
        
        For OpenSim models, we typically scale based on:
        - Height scaling: ratio of subject height to model height
        - Mass scaling: ratio of subject mass to model mass
        
        Args:
            subject_data: Dictionary containing subject anthropometric data
            
        Returns:
            Dictionary with scaling factors
        """
        # Default model parameters (from Hamner2010 model)
        default_height = 1.75  # meters (average adult height)
        default_mass = 75.0    # kg (average adult mass)
        
        subject_height = float(subject_data['Height'])
        subject_mass = float(subject_data['Weight'])
        
        # Calculate scaling factors
        height_scale = subject_height / default_height
        mass_scale = subject_mass / default_mass
        
        # For most OpenSim scaling, we use uniform scaling based on height
        # Mass scaling is typically handled separately in the model
        uniform_scale = height_scale
        
        scaling_factors = {
            'uniform_scale': uniform_scale,
            'height_scale': height_scale,
            'mass_scale': mass_scale,
            'subject_height': subject_height,
            'subject_mass': subject_mass
        }
        
        logger.info(f"Scaling factors for {subject_data['Subject']}:")
        logger.info(f"  Height: {subject_height:.3f}m (scale: {height_scale:.3f})")
        logger.info(f"  Mass: {subject_mass:.1f}kg (scale: {mass_scale:.3f})")
        logger.info(f"  Uniform scale: {uniform_scale:.3f}")
        
        return scaling_factors
    
    def scale_model(self, subject_id: str, scaling_factors: Dict[str, float]) -> str:
        """
        Create a scaled version of the OpenSim model for the subject.
        
        Args:
            subject_id: Subject ID
            scaling_factors: Dictionary containing scaling factors
            
        Returns:
            Path to the scaled model file
        """
        try:
            # Parse the original model
            tree = ET.parse(self.model_file)
            root = tree.getroot()
            
            # Update model name
            model_element = root.find('Model')
            if model_element is not None:
                safe_name = self._sanitize_component_name(f'GaitModel_{subject_id}')
                model_element.set('name', safe_name)
            
            # Scale body masses
            self._scale_body_masses(root, scaling_factors['mass_scale'])
            
            # Scale body dimensions (lengths, positions)
            self._scale_body_dimensions(root, scaling_factors['uniform_scale'])
            
            # Create output directory (standardized as 'opensim')
            output_dir = self.dataset_dir / subject_id / "opensim"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save scaled model
            output_file = output_dir / f"{subject_id}.osim"
            tree.write(output_file, encoding='UTF-8', xml_declaration=True)
            
            logger.info(f"Created scaled model: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error scaling model for {subject_id}: {e}")
            raise

    def _sanitize_component_name(self, name: str) -> str:
        """Sanitize OpenSim component names to avoid invalid characters.
        Allows letters, digits, and underscores only.
        """
        # Replace invalid characters with underscore
        name = re.sub(r"[^A-Za-z0-9_]", "_", name)
        # Ensure it does not start with a digit
        if re.match(r"^[0-9]", name):
            name = f"_{name}"
        return name
    
    def _scale_body_masses(self, root: ET.Element, mass_scale: float) -> None:
        """
        Scale body masses in the model.
        
        Args:
            root: Root element of the XML tree
            mass_scale: Mass scaling factor
        """
        # Find all Body elements and scale their masses
        for body in root.findall('.//Body'):
            mass_element = body.find('mass')
            if mass_element is not None:
                try:
                    original_mass = float(mass_element.text)
                    scaled_mass = original_mass * mass_scale
                    mass_element.text = f"{scaled_mass:.6f}"
                    logger.debug(f"Scaled body mass: {body.get('name', 'unnamed')} "
                               f"{original_mass:.3f} -> {scaled_mass:.3f} kg")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not scale mass for body {body.get('name', 'unnamed')}: {e}")
    
    def _scale_body_dimensions(self, root: ET.Element, uniform_scale: float) -> None:
        """
        Scale body dimensions (positions, lengths) in the model.
        
        Args:
            root: Root element of the XML tree
            uniform_scale: Uniform scaling factor
        """
        # Scale joint positions
        for joint in root.findall('.//Joint'):
            self._scale_joint_positions(joint, uniform_scale)
        
        # Scale body positions
        for body in root.findall('.//Body'):
            self._scale_body_positions(body, uniform_scale)
        
        # Scale muscle path points
        for muscle in root.findall('.//Thelen2003Muscle'):
            self._scale_muscle_paths(muscle, uniform_scale)
        
        # Scale marker positions
        for marker in root.findall('.//Marker'):
            self._scale_marker_positions(marker, uniform_scale)
    
    def _scale_joint_positions(self, joint: ET.Element, scale: float) -> None:
        """Scale joint position coordinates."""
        for location in joint.findall('.//location'):
            try:
                coords = [float(x) for x in location.text.split()]
                scaled_coords = [coord * scale for coord in coords]
                location.text = ' '.join([f"{coord:.6f}" for coord in scaled_coords])
            except (ValueError, AttributeError):
                pass
    
    def _scale_body_positions(self, body: ET.Element, scale: float) -> None:
        """Scale body position coordinates."""
        for location in body.findall('.//location'):
            try:
                coords = [float(x) for x in location.text.split()]
                scaled_coords = [coord * scale for coord in coords]
                location.text = ' '.join([f"{coord:.6f}" for coord in scaled_coords])
            except (ValueError, AttributeError):
                pass
    
    def _scale_muscle_paths(self, muscle: ET.Element, scale: float) -> None:
        """Scale muscle path point coordinates."""
        for path_point in muscle.findall('.//PathPoint'):
            for location in path_point.findall('.//location'):
                try:
                    coords = [float(x) for x in location.text.split()]
                    scaled_coords = [coord * scale for coord in coords]
                    location.text = ' '.join([f"{coord:.6f}" for coord in scaled_coords])
                except (ValueError, AttributeError):
                    pass
    
    def _scale_marker_positions(self, marker: ET.Element, scale: float) -> None:
        """Scale marker position coordinates."""
        for location in marker.findall('.//location'):
            try:
                coords = [float(x) for x in location.text.split()]
                scaled_coords = [coord * scale for coord in coords]
                location.text = ' '.join([f"{coord:.6f}" for coord in scaled_coords])
            except (ValueError, AttributeError):
                pass
    
    def generate_models_for_all_subjects(self) -> None:
        """
        Generate scaled models for all subjects in the dataset.
        """
        logger.info("Starting model generation for all subjects...")
        
        # Load subject information
        subject_df = self.load_subject_info()
        
        # Get list of subjects that exist in the dataset
        existing_subjects = []
        for subject_dir in self.dataset_dir.iterdir():
            if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
                continue
            name = subject_dir.name
            # Accept any alphanumeric subject folder (e.g., AB06, K01, S001)
            if any(ch.isalnum() for ch in name):
                existing_subjects.append(name)
        
        logger.info(f"Found {len(existing_subjects)} subject directories: {existing_subjects}")
        
        successful_generations = 0
        failed_generations = 0
        
        for subject_id in sorted(existing_subjects):
            try:
                logger.info(f"Processing subject: {subject_id}")
                
                # Get subject data
                subject_data = self.get_subject_data(subject_id, subject_df)
                if subject_data is None:
                    logger.warning(f"Skipping {subject_id} - no data in SubjectInfo.csv")
                    failed_generations += 1
                    continue
                
                # Calculate scaling factors
                scaling_factors = self.calculate_scaling_factors(subject_data)
                
                # Generate scaled model
                model_file = self.scale_model(subject_id, scaling_factors)
                
                logger.info(f"✅ Successfully generated model for {subject_id}")
                successful_generations += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to generate model for {subject_id}: {e}")
                failed_generations += 1
        
        logger.info(f"Model generation complete:")
        logger.info(f"  ✅ Successful: {successful_generations}")
        logger.info(f"  ❌ Failed: {failed_generations}")
        logger.info(f"  📁 Total subjects: {len(existing_subjects)}")


def main():
    """Main function to run the model generation script."""
    parser = argparse.ArgumentParser(
        description="Generate scaled OpenSim models for subjects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_scaled_models.py "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Keaton" \
    --model-file "/Users/luorix/Documents/OpenSim/Resources/Models/Gait2392_SimBody/gait2392_simbody.osim"
  python generate_scaled_models.py "/path/to/dataset" --model-file "/path/to/model.osim"
        """
    )
    
    parser.add_argument(
        'dataset_directory',
        help='Path to the dataset directory containing SubjectInfo.csv'
    )
    
    parser.add_argument(
        '--model-file',
        default='/Users/luorix/Documents/OpenSim/Resources/Models/Gait2392_SimBody/gait2392_simbody.osim',
        help='Path to the generic OpenSim model file (default: gait2392_simbody.osim)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize model generator
        generator = OpenSimModelGenerator(args.dataset_directory, args.model_file)
        
        # Generate models for all subjects
        generator.generate_models_for_all_subjects()
        
        logger.info("🎉 Model generation completed successfully!")
        
    except Exception as e:
        logger.error(f"💥 Model generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
