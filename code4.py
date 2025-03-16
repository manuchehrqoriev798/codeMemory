"""
Machine Learning Utilities Module

This module provides utilities for machine learning,
including preprocessing, model training, and evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional, Any, Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class for preprocessing data for machine learning."""
    
    def __init__(self):
        """Initialize preprocessor with empty transformers."""
        self.numerical_scaler = None
        self.categorical_encoders = {}
        logger.info("DataPreprocessor initialized")
    
    def fit_transform_numerical(self, X: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Fit and transform numerical features."""
        if method == 'standard':
            self.numerical_scaler = StandardScaler()
        elif method == 'minmax':
            self.numerical_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        X_copy = X.copy()
        
        if len(numerical_columns) > 0:
            X_copy[numerical_columns] = self.numerical_scaler.fit_transform(X[numerical_columns])
            logger.info("Fitted and transformed %d numerical columns using %s scaling", 
                      len(numerical_columns), method)
        else:
            logger.warning("No numerical columns found in the dataset")
        
        return X_copy
    
    def transform_numerical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform numerical features using fitted scaler."""
        if self.numerical_scaler is None:
            raise ValueError("Numerical scaler not fitted. Call fit_transform_numerical first.")
        
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        X_copy = X.copy()
        
        if len(numerical_columns) > 0:
            X_copy[numerical_columns] = self.numerical_scaler.transform(X[numerical_columns])
        
        return X_copy
    
    def fit_transform_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform categorical features using label encoding."""
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        X_copy = X.copy()
        
        for column in categorical_columns:
            encoder = LabelEncoder()
            X_copy[column] = encoder.fit_transform(X[column])
            self.categorical_encoders[column] = encoder
        
        logger.info("Fitted and transformed %d categorical columns", len(categorical_columns))
        return X_copy
    
    def transform_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features using fitted encoders."""
        X_copy = X.copy()
        
        for column, encoder in self.categorical_encoders.items():
            if column in X.columns:
                X_copy[column] = encoder.transform(X[column])
        
        return X_copy
    
    def preprocess_data(self, X: pd.DataFrame, 
                       numerical_method: str = 'standard', 
                       fit: bool = True) -> pd.DataFrame:
        """Preprocess all data (numerical and categorical)."""
        X_copy = X.copy()
        
        if fit:
            X_copy = self.fit_transform_numerical(X_copy, method=numerical_method)
            X_copy = self.fit_transform_categorical(X_copy)
        else:
            X_copy = self.transform_numerical(X_copy)
            X_copy = self.transform_categorical(X_copy)
        
        return X_copy
    
    def save(self, path: str) -> None:
        """Save the preprocessor to disk."""
        joblib.dump(self, path)
        logger.info("Preprocessor saved to %s", path)
    
    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """Load a preprocessor from disk."""
        preprocessor = joblib.load(path)
        logger.info("Preprocessor loaded from %s", path)
        return preprocessor

class ModelTrainer:
    """Class for training machine learning models."""
    
    def __init__(self, model_type: str, params: Optional[Dict[str, Any]] = None):
        """Initialize with model type and parameters."""
        self.model_type = model_type
        self.params = params or {}
        self.model = self._create_model()
        logger.info("ModelTrainer initialized with model type: %s", model_type)
    
    def _create_model(self):
        """Create a model instance based on model_type."""
        if self.model_type == 'logistic_regression':
            return LogisticRegression(**self.params)
        elif self.model_type == 'random_forest_classifier':
            return RandomForestClassifier(**self.params)
        elif self.model_type == 'svm_classifier':
            return SVC(**self.params)
        elif self.model_type == 'linear_regression':
            return LinearRegression(**self.params)
        elif self.model_type == 'random_forest_regressor':
            return RandomForestRegressor(**self.params)
        elif self.model_type == 'svm_regressor':
            return SVR(**self.params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model on the given data."""
        logger.info("Training %s model on %d samples", self.model_type, len(X_train))
        self.model.fit(X_train, y_train)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def evaluate_classifier(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate a classification model."""
        if not hasattr(self.model, "predict_proba"):
            logger.warning("Model doesn't support predict_proba, using basic metrics only")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        logger.info("Classification metrics: %s", metrics)
        return metrics
    
    def evaluate_regressor(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate a regression model."""
        y_pred = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info("Regression metrics: %s", metrics)
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                     cv: int = 5, scoring: str = 'accuracy') -> Dict[str, float]:
        """Perform cross-validation."""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        metrics = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score': scores.max()
        }
        
        logger.info("Cross-validation %s: mean=%.4f, std=%.4f", 
                  scoring, metrics['mean_score'], metrics['std_score'])
        return metrics
    
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                           param_grid: Dict[str, List[Any]], 
                           cv: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """Find optimal hyperparameters using grid search."""
        grid_search = GridSearchCV(self.model, param_grid, cv=cv, scoring=scoring)
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
        
        logger.info("Best parameters: %s", results['best_params'])
        logger.info("Best score: %.4f", results['best_score'])
        return results
    
    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        joblib.dump(self.model, path)
        logger.info("Model saved to %s", path)
    
    @classmethod
    def load_model(cls, path: str) -> 'ModelTrainer':
        """Load a trained model from disk."""
        model = joblib.load(path)
        model_type = model.__class__.__name__
        trainer = cls(model_type)
        trainer.model = model
        logger.info("Model loaded from %s", path)
        return trainer

def prepare_and_split_data(df: pd.DataFrame, target_column: str, 
                         test_size: float = 0.2, 
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare and split data into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info("Data split into training (%d samples) and testing (%d samples) sets", 
              len(X_train), len(X_test))
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                           y_train: pd.Series, y_test: pd.Series,
                           model_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Train and evaluate a model (wrapper function)."""
    trainer = ModelTrainer(model_type, params)
    trainer.train(X_train, y_train)
    
    # Evaluate based on task type
    if model_type in ['logistic_regression', 'random_forest_classifier', 'svm_classifier']:
        metrics = trainer.evaluate_classifier(X_test, y_test)
    else:
        metrics = trainer.evaluate_regressor(X_test, y_test)
    
    return {
        'trainer': trainer,
        'metrics': metrics
    }

def main():
    """Demonstrate the use of ML utilities."""
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 200),
        'feature2': np.random.normal(0, 1, 200),
        'feature3': np.random.choice(['A', 'B', 'C'], 200),
        'target': np.random.randint(0, 2, 200)  # Binary classification
    })
    
    # Prepare and split data
    X_train, X_test, y_train, y_test = prepare_and_split_data(df, 'target')
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X_train_processed = preprocessor.preprocess_data(X_train)
    X_test_processed = preprocessor.preprocess_data(X_test, fit=False)
    
    # Train and evaluate a model
    result = train_and_evaluate_model(
        X_train_processed, X_test_processed, y_train, y_test,
        model_type='random_forest_classifier',
        params={'n_estimators': 100, 'random_state': 42}
    )
    
    print(f"Model evaluation metrics: {result['metrics']}")
    
    # Save the model and preprocessor
    result['trainer'].save_model('random_forest_model.joblib')
    preprocessor.save('preprocessor.joblib')
    
    print("Model and preprocessor saved.")

if __name__ == "__main__":
    main() 