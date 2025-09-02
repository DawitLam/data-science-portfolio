"""
Model training pipeline for fracture risk prediction.

This module provides the main training pipeline with multiple algorithms,
hyperparameter optimization, and comprehensive evaluation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    average_precision_score, brier_score_loss, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import optuna

# Custom modules
try:
    from src.data.load_data import DataLoader
    from src.features.build_features import MedicalFeatureEngineer
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append('.')
    from src.data.load_data import DataLoader
    from src.features.build_features import MedicalFeatureEngineer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FractureRiskModelTrainer:
    """Main class for training fracture risk prediction models."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.results = {}
        
        # Create output directories
        self.model_dir = Path("models/trained_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load and prepare training data."""
        logger.info("Loading and preparing data...")
        
        # Load data
        loader = DataLoader("config/feature_config.yaml")
        # Pass model config to loader for training parameters
        loader.model_config = self.config
        X, y = loader.load_training_data(
            include_surveys=True,
            include_fracture_details=False
        )
        
        # Apply feature engineering
        feature_config = loader.config
        engineer = MedicalFeatureEngineer(feature_config)
        X_engineered = engineer.apply_all_transformations(X)
        
        # Create train/validation split
        X_train, X_val, _, y_train, y_val, _ = loader.create_train_val_test_split(
            X_engineered, y
        )
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        logger.info(f"Data prepared: {X_train.shape[0]} train, {X_val.shape[0]} validation samples")
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
        
        return X_train, X_val, y_train, y_val
    
    def preprocess_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess features for model training."""
        logger.info("Preprocessing features...")
        
        # Handle categorical variables
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        
        X_train_processed = X_train.copy()
        X_val_processed = X_val.copy()
        
        # Simple encoding for categorical variables
        for col in categorical_cols:
            # Convert to string first to handle categorical types
            X_train_processed[col] = X_train_processed[col].astype(str)
            X_val_processed[col] = X_val_processed[col].astype(str)
            
            # Label encoding for tree-based models
            unique_vals = pd.concat([X_train_processed[col], X_val_processed[col]]).unique()
            val_map = {val: i for i, val in enumerate(unique_vals)}
            
            X_train_processed[col] = X_train_processed[col].map(val_map).fillna(-1)
            X_val_processed[col] = X_val_processed[col].map(val_map).fillna(-1)
        
        # Scale numerical features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_processed[numerical_cols])
        X_val_scaled = scaler.transform(X_val_processed[numerical_cols])
        
        # Combine processed features
        X_train_final = X_train_processed.copy()
        X_val_final = X_val_processed.copy()
        X_train_final[numerical_cols] = X_train_scaled
        X_val_final[numerical_cols] = X_val_scaled
        
        # Store scaler
        self.scalers['standard_scaler'] = scaler
        
        # Fill any remaining NaN values
        X_train_final = X_train_final.fillna(0)
        X_val_final = X_val_final.fillna(0)
        
        logger.info("Feature preprocessing complete")
        return X_train_final.values, X_val_final.values
    
    def train_logistic_regression(self, X_train: np.ndarray, X_val: np.ndarray, 
                                 y_train: pd.Series, y_val: pd.Series) -> Dict:
        """Train Logistic Regression model."""
        logger.info("Training Logistic Regression...")
        
        config = self.config['models']['logistic_regression']
        model = LogisticRegression(**config)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = self._evaluate_model(model, X_train, X_val, y_train, y_val)
        
        # Store model
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = metrics
        
        logger.info(f"Logistic Regression - Validation AUC: {metrics['val_auc']:.4f}")
        return metrics
    
    def train_random_forest(self, X_train: np.ndarray, X_val: np.ndarray,
                           y_train: pd.Series, y_val: pd.Series) -> Dict:
        """Train Random Forest model."""
        logger.info("Training Random Forest...")
        
        config = self.config['models']['random_forest']
        model = RandomForestClassifier(**config)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = self._evaluate_model(model, X_train, X_val, y_train, y_val)
        
        # Add feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metrics['feature_importance'] = feature_importance.to_dict('records')
        
        # Store model
        self.models['random_forest'] = model
        self.results['random_forest'] = metrics
        
        logger.info(f"Random Forest - Validation AUC: {metrics['val_auc']:.4f}")
        return metrics
    
    def train_xgboost(self, X_train: np.ndarray, X_val: np.ndarray,
                     y_train: pd.Series, y_val: pd.Series) -> Dict:
        """Train XGBoost model."""
        logger.info("Training XGBoost...")
        
        config = self.config['models']['xgboost']
        model = xgb.XGBClassifier(**config)
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        metrics = self._evaluate_model(model, X_train, X_val, y_train, y_val)
        
        # Add feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metrics['feature_importance'] = feature_importance.to_dict('records')
        
        # Store model
        self.models['xgboost'] = model
        self.results['xgboost'] = metrics
        
        logger.info(f"XGBoost - Validation AUC: {metrics['val_auc']:.4f}")
        return metrics
    
    def train_lightgbm(self, X_train: np.ndarray, X_val: np.ndarray,
                      y_train: pd.Series, y_val: pd.Series) -> Dict:
        """Train LightGBM model."""
        logger.info("Training LightGBM...")
        
        config = self.config['models']['lightgbm']
        model = lgb.LGBMClassifier(**config)
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Evaluate
        metrics = self._evaluate_model(model, X_train, X_val, y_train, y_val)
        
        # Add feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metrics['feature_importance'] = feature_importance.to_dict('records')
        
        # Store model
        self.models['lightgbm'] = model
        self.results['lightgbm'] = metrics
        
        logger.info(f"LightGBM - Validation AUC: {metrics['val_auc']:.4f}")
        return metrics
    
    def _evaluate_model(self, model, X_train: np.ndarray, X_val: np.ndarray,
                       y_train: pd.Series, y_val: pd.Series) -> Dict:
        """Comprehensive model evaluation."""
        # Predictions
        y_train_pred = model.predict_proba(X_train)[:, 1]
        y_val_pred = model.predict_proba(X_val)[:, 1]
        
        y_train_pred_binary = model.predict(X_train)
        y_val_pred_binary = model.predict(X_val)
        
        # Metrics
        metrics = {
            # Training metrics
            'train_auc': roc_auc_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred_binary),
            'train_recall': recall_score(y_train, y_train_pred_binary),
            'train_f1': f1_score(y_train, y_train_pred_binary),
            
            # Validation metrics
            'val_auc': roc_auc_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred_binary),
            'val_recall': recall_score(y_val, y_val_pred_binary),
            'val_f1': f1_score(y_val, y_val_pred_binary),
            'val_average_precision': average_precision_score(y_val, y_val_pred),
            'val_brier_score': brier_score_loss(y_val, y_val_pred),
            
            # Additional metrics
            'overfitting': roc_auc_score(y_train, y_train_pred) - roc_auc_score(y_val, y_val_pred)
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        metrics['cv_auc_mean'] = cv_scores.mean()
        metrics['cv_auc_std'] = cv_scores.std()
        
        return metrics
    
    def train_all_models(self) -> Dict:
        """Train all configured models."""
        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("=" * 60)
        
        # Prepare data
        X_train, X_val, y_train, y_val = self.prepare_data()
        
        # Preprocess features
        X_train_processed, X_val_processed = self.preprocess_features(X_train, X_val)
        
        # Train models
        model_results = {}
        
        # Logistic Regression
        model_results['logistic_regression'] = self.train_logistic_regression(
            X_train_processed, X_val_processed, y_train, y_val
        )
        
        # Random Forest
        model_results['random_forest'] = self.train_random_forest(
            X_train_processed, X_val_processed, y_train, y_val
        )
        
        # XGBoost
        model_results['xgboost'] = self.train_xgboost(
            X_train_processed, X_val_processed, y_train, y_val
        )
        
        # LightGBM  
        model_results['lightgbm'] = self.train_lightgbm(
            X_train_processed, X_val_processed, y_train, y_val
        )
        
        # Select best model
        best_model_name = self._select_best_model()
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best validation AUC: {self.results[best_model_name]['val_auc']:.4f}")
        
        return model_results
    
    def _select_best_model(self) -> str:
        """Select best performing model based on validation metrics."""
        metric = self.config['selection']['metric']
        threshold = self.config['selection']['threshold']
        
        # Find best model by primary metric
        best_score = -np.inf
        best_model = None
        
        for model_name, results in self.results.items():
            score = results[f'val_{metric}']
            if score > best_score and score >= threshold:
                best_score = score
                best_model = model_name
        
        if best_model is None:
            # Fallback to highest scoring model even if below threshold
            best_model = max(self.results.keys(), 
                           key=lambda x: self.results[x][f'val_{metric}'])
            logger.warning(f"No model met threshold {threshold}, selecting best: {best_model}")
        
        return best_model
    
    def save_models(self) -> None:
        """Save trained models and artifacts."""
        logger.info("Saving models and artifacts...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all models
        for model_name, model in self.models.items():
            model_path = self.model_dir / f"{model_name}_{timestamp}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save best model as 'best_model.pkl'
        best_model_name = self._select_best_model()
        best_model_path = self.model_dir / "best_model.pkl"
        joblib.dump(self.models[best_model_name], best_model_path)
        
        # Save scaler
        scaler_path = self.model_dir / "scaler.pkl"
        joblib.dump(self.scalers['standard_scaler'], scaler_path)
        
        # Save feature names
        feature_names_path = self.model_dir / "feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save results
        results_path = self.model_dir / f"training_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info("All artifacts saved successfully")
    
    def print_model_comparison(self) -> None:
        """Print comparison of all trained models."""
        logger.info("=" * 80)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 80)
        
        comparison_df = pd.DataFrame(self.results).T
        
        # Select key metrics for comparison
        key_metrics = ['val_auc', 'val_precision', 'val_recall', 'val_f1', 'overfitting']
        
        print(comparison_df[key_metrics].round(4))
        
        # Highlight best model for each metric
        for metric in key_metrics:
            best_model = comparison_df[metric].idxmax()
            best_score = comparison_df[metric].max()
            print(f"\nBest {metric}: {best_model} ({best_score:.4f})")


def train_fracture_risk_model(algorithm: str = 'xgboost') -> Tuple[Any, Dict]:
    """
    Convenience function to train a single model.
    
    Args:
        algorithm: Model algorithm to train
        
    Returns:
        Tuple: (trained_model, metrics)
    """
    trainer = FractureRiskModelTrainer()
    X_train, X_val, y_train, y_val = trainer.prepare_data()
    X_train_processed, X_val_processed = trainer.preprocess_features(X_train, X_val)
    
    if algorithm == 'xgboost':
        metrics = trainer.train_xgboost(X_train_processed, X_val_processed, y_train, y_val)
        model = trainer.models['xgboost']
    elif algorithm == 'lightgbm':
        metrics = trainer.train_lightgbm(X_train_processed, X_val_processed, y_train, y_val)
        model = trainer.models['lightgbm']
    elif algorithm == 'random_forest':
        metrics = trainer.train_random_forest(X_train_processed, X_val_processed, y_train, y_val)
        model = trainer.models['random_forest']
    elif algorithm == 'logistic_regression':
        metrics = trainer.train_logistic_regression(X_train_processed, X_val_processed, y_train, y_val)
        model = trainer.models['logistic_regression']
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return model, metrics


if __name__ == "__main__":
    # Train all models
    trainer = FractureRiskModelTrainer()
    results = trainer.train_all_models()
    
    # Print comparison
    trainer.print_model_comparison()
    
    # Save models
    trainer.save_models()
