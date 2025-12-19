"""Machine learning model utilities for AI Lab Studio."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from typing import Dict, Any, Tuple, Optional
import joblib
from datetime import datetime
import os

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class MLModelTrainer:
    """Handles machine learning model training and evaluation."""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.task_type = None  # 'classification' or 'regression'
        self.feature_names = None
        self.target_name = None
        
    def create_model(self, algorithm: str, task_type: str, params: Dict[str, Any]):
        """Create a machine learning model based on algorithm and parameters.
        
        Args:
            algorithm: Name of the algorithm
            task_type: 'classification' or 'regression'
            params: Dictionary of hyperparameters
            
        Returns:
            model: Created model instance
        """
        self.task_type = task_type
        self.model_type = algorithm
        
        if task_type == 'classification':
            if algorithm == 'Logistic Regression':
                self.model = LogisticRegression(
                    C=params.get('C', 1.0),
                    max_iter=params.get('max_iter', 1000),
                    random_state=42
                )
            elif algorithm == 'Random Forest':
                self.model = RandomForestClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', None),
                    min_samples_split=params.get('min_samples_split', 2),
                    random_state=42
                )
            elif algorithm == 'XGBoost' and XGBOOST_AVAILABLE:
                self.model = XGBClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 6),
                    learning_rate=params.get('learning_rate', 0.1),
                    random_state=42
                )
            elif algorithm == 'SVM':
                self.model = SVC(
                    C=params.get('C', 1.0),
                    kernel=params.get('kernel', 'rbf'),
                    probability=True,
                    random_state=42
                )
        elif task_type == 'regression':
            if algorithm == 'Linear Regression':
                self.model = LinearRegression()
            elif algorithm == 'Random Forest':
                self.model = RandomForestRegressor(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', None),
                    min_samples_split=params.get('min_samples_split', 2),
                    random_state=42
                )
            elif algorithm == 'XGBoost' and XGBOOST_AVAILABLE:
                self.model = XGBRegressor(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 6),
                    learning_rate=params.get('learning_rate', 0.1),
                    random_state=42
                )
            elif algorithm == 'SVM':
                self.model = SVR(
                    C=params.get('C', 1.0),
                    kernel=params.get('kernel', 'rbf')
                )
        
        return self.model
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   feature_names: Optional[list] = None):
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            feature_names: List of feature names
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        self.feature_names = feature_names or X_train.columns.tolist()
        self.target_name = y_train.name
        
        self.model.fit(X_train, y_train)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (classification only).
        
        Args:
            X: Features for prediction
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model does not support probability predictions.")
    
    def evaluate_classification(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate classification model.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        # Handle multiclass vs binary
        average_method = 'weighted' if len(np.unique(y_test)) > 2 else 'binary'
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average=average_method, zero_division=0),
            'recall': recall_score(y_test, y_pred, average=average_method, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average=average_method, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }
        
        return metrics
    
    def evaluate_regression(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate regression model.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance for tree-based models.
        
        Returns:
            pd.DataFrame: Feature importance dataframe or None
        """
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None
    
    def save_model(self, model_name: str, models_dir: str = 'models',
                  metrics: Optional[Dict] = None) -> str:
        """Save model to disk.
        
        Args:
            model_name: Name for the saved model
            models_dir: Directory to save models
            metrics: Model performance metrics
            
        Returns:
            str: Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(models_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = os.path.join(models_dir, filename)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'task_type': self.task_type,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'metrics': metrics,
            'timestamp': timestamp
        }
        
        joblib.dump(model_data, filepath)
        return filepath
    
    def load_model(self, filepath: str):
        """Load model from disk.
        
        Args:
            filepath: Path to saved model file
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.task_type = model_data['task_type']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data.get('target_name')
        
        return model_data.get('metrics', {})


def get_model_info(filepath: str) -> Dict[str, Any]:
    """Get information about a saved model without fully loading it.
    
    Args:
        filepath: Path to saved model file
        
    Returns:
        dict: Model information
    """
    model_data = joblib.load(filepath)
    
    info = {
        'model_type': model_data.get('model_type', 'Unknown'),
        'task_type': model_data.get('task_type', 'Unknown'),
        'feature_names': model_data.get('feature_names', []),
        'target_name': model_data.get('target_name', 'Unknown'),
        'metrics': model_data.get('metrics', {}),
        'timestamp': model_data.get('timestamp', 'Unknown')
    }
    
    return info