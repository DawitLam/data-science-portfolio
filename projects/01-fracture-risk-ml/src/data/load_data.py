"""
Data loading utilities for fracture risk prediction project.

This module provides functions to load and prepare data for model training
and prediction. Designed to work with both synthetic (public) and real (private) datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import yaml
import logging
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Data loading and preparation utilities."""
    
    def __init__(self, config_path: str = "config/feature_config.yaml"):
        """Initialize data loader with configuration."""
        self.config = self._load_config(config_path)
        self.data_path = Path(self.config['data']['raw_data_path'])
        self.model_config = None  # Will be set by model trainer if needed
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_master_data(self) -> pd.DataFrame:
        """
        Load master patient dataset.
        
        Returns:
            pd.DataFrame: Master patient data with all features
        """
        file_path = self.data_path / "master_patient_data.csv"
        
        if not file_path.exists():
            logger.error(f"Master data file not found: {file_path}")
            raise FileNotFoundError(f"Master data file not found: {file_path}")
        
        logger.info(f"Loading master data from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Basic data validation
        self._validate_master_data(df)
        
        logger.info(f"Loaded {len(df):,} patient records")
        return df
    
    def load_fracture_events(self) -> pd.DataFrame:
        """
        Load fracture events dataset.
        
        Returns:
            pd.DataFrame: Fracture events data
        """
        file_path = self.data_path / "fracture_events.csv"
        
        if not file_path.exists():
            logger.warning(f"Fracture events file not found: {file_path}")
            return pd.DataFrame()
        
        logger.info(f"Loading fracture events from: {file_path}")
        df = pd.read_csv(file_path)
        
        logger.info(f"Loaded {len(df):,} fracture events")
        return df
    
    def load_patient_surveys(self) -> pd.DataFrame:
        """
        Load patient survey data.
        
        Returns:
            pd.DataFrame: Patient survey responses
        """
        file_path = self.data_path / "patient_surveys.csv"
        
        if not file_path.exists():
            logger.warning(f"Patient surveys file not found: {file_path}")
            return pd.DataFrame()
        
        logger.info(f"Loading patient surveys from: {file_path}")
        df = pd.read_csv(file_path)
        
        logger.info(f"Loaded {len(df):,} survey responses")
        return df
    
    def _validate_master_data(self, df: pd.DataFrame) -> None:
        """Validate master data quality."""
        required_cols = self.config['quality_checks']['required_columns']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check data ranges
        range_checks = self.config['quality_checks']['range_validation']
        for col, (min_val, max_val) in range_checks.items():
            if col in df.columns:
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if out_of_range > 0:
                    logger.warning(f"{col}: {out_of_range} values out of range [{min_val}, {max_val}]")
    
    def load_training_data(self, 
                          include_surveys: bool = True,
                          include_fracture_details: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare training dataset.
        
        Args:
            include_surveys: Whether to include patient survey data
            include_fracture_details: Whether to include detailed fracture event data
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)
        """
        # Load master data
        master_data = self.load_master_data()
        
        # Optionally merge with survey data
        if include_surveys:
            surveys = self.load_patient_surveys()
            if not surveys.empty:
                master_data = master_data.merge(
                    surveys, 
                    on='patient_id', 
                    how='left'
                )
                logger.info("Merged survey data")
        
        # Optionally add fracture event details
        if include_fracture_details:
            fractures = self.load_fracture_events()
            if not fractures.empty:
                # Aggregate fracture data per patient
                fracture_agg = self._aggregate_fracture_data(fractures)
                master_data = master_data.merge(
                    fracture_agg,
                    on='patient_id',
                    how='left'
                )
                logger.info("Merged fracture event data")
        
        # Separate features and target
        target_col = self.config['target']['column']
        
        if target_col not in master_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        X = master_data.drop(columns=[target_col, 'patient_id'])
        y = master_data[target_col]
        
        logger.info(f"Prepared training data: {X.shape[0]:,} samples, {X.shape[1]} features")
        return X, y
    
    def _aggregate_fracture_data(self, fractures: pd.DataFrame) -> pd.DataFrame:
        """Aggregate fracture event data per patient."""
        agg_data = fractures.groupby('patient_id').agg({
            'fracture_site': lambda x: '; '.join(x.unique()),
            'severity': lambda x: max(x, key=lambda v: ['Minor', 'Moderate', 'Severe'].index(v)),
            'recovery_days': 'mean',
            'mechanism': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'
        }).reset_index()
        
        # Rename columns for clarity
        agg_data.columns = [
            'patient_id', 
            'fracture_sites', 
            'worst_fracture_severity',
            'avg_recovery_days',
            'most_common_mechanism'
        ]
        
        return agg_data
    
    def create_train_val_test_split(self, 
                                   X: pd.DataFrame, 
                                   y: pd.Series,
                                   test_size: float = None,
                                   val_size: float = None,
                                   random_state: int = None) -> Tuple[pd.DataFrame, ...]:
        """
        Create train/validation/test splits.
        
        Args:
            X: Features
            y: Target
            test_size: Test set proportion
            val_size: Validation set proportion  
            random_state: Random seed
            
        Returns:
            Tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Use config defaults if not provided
        config = self.model_config['training'] if self.model_config else {
            'test_size': 0.2, 'validation_size': 0.2, 'random_state': 42
        }
        test_size = test_size or config['test_size']
        val_size = val_size or config['validation_size'] 
        random_state = random_state or config['random_state']
        stratify = y if config['stratify'] else None
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        stratify_temp = y_temp if config['stratify'] else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_temp
        )
        
        logger.info(f"Created data splits:")
        logger.info(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(f"  Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
        logger.info(f"  Test: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def load_training_data(config_path: str = "config/feature_config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Convenience function to load training data with train/val split.
    
    Args:
        config_path: Path to feature configuration file
        
    Returns:
        Tuple: X_train, X_val, y_train, y_val
    """
    loader = DataLoader(config_path)
    X, y = loader.load_training_data()
    X_train, X_val, _, y_train, y_val, _ = loader.create_train_val_test_split(X, y)
    
    return X_train, X_val, y_train, y_val


def load_full_dataset(config_path: str = "config/feature_config.yaml") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to load full dataset.
    
    Args:
        config_path: Path to feature configuration file
        
    Returns:
        Tuple: X (features), y (target)
    """
    loader = DataLoader(config_path)
    return loader.load_training_data()


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Load individual datasets
    master_data = loader.load_master_data()
    print(f"Master data shape: {master_data.shape}")
    
    # Load training data
    X, y = loader.load_training_data()
    print(f"Training data shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Create splits
    X_train, X_val, X_test, y_train, y_val, y_test = loader.create_train_val_test_split(X, y)
