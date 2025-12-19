"""Data processing utilities for AI Lab Studio."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from typing import Tuple, Optional, List
import streamlit as st


class DataProcessor:
    """Handles all data processing operations."""
    
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        
    @staticmethod
    def load_data(uploaded_file) -> pd.DataFrame:
        """Load data from uploaded CSV file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pd.DataFrame: Loaded dataframe
            
        Raises:
            Exception: If file cannot be loaded
        """
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    @staticmethod
    def get_data_info(df: pd.DataFrame) -> dict:
        """Get basic information about the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            dict: Dictionary containing data information
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        return info
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', 
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Handle missing values in the dataframe.
        
        Args:
            df: Input dataframe
            strategy: Strategy to handle missing values ('drop', 'mean', 'median', 'mode')
            columns: Specific columns to apply strategy (if None, applies to all)
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.columns.tolist()
        
        if strategy == 'drop':
            df_copy = df_copy.dropna(subset=columns)
        elif strategy == 'mean':
            for col in columns:
                if df_copy[col].dtype in ['float64', 'int64']:
                    df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in columns:
                if df_copy[col].dtype in ['float64', 'int64']:
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in columns:
                if len(df_copy[col].mode()) > 0:
                    df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        
        return df_copy
    
    def encode_categorical(self, df: pd.DataFrame, method: str = 'label',
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Encode categorical variables.
        
        Args:
            df: Input dataframe
            method: Encoding method ('label' or 'onehot')
            columns: Specific columns to encode (if None, encodes all object columns)
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.select_dtypes(include=['object']).columns.tolist()
        
        if method == 'label':
            for col in columns:
                if col in df_copy.columns:
                    le = LabelEncoder()
                    df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                    self.label_encoders[col] = le
        elif method == 'onehot':
            df_copy = pd.get_dummies(df_copy, columns=columns, drop_first=True)
        
        return df_copy
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard',
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Scale numerical features.
        
        Args:
            df: Input dataframe
            method: Scaling method ('standard' or 'minmax')
            columns: Specific columns to scale (if None, scales all numerical columns)
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if len(columns) == 0:
            return df_copy
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        
        df_copy[columns] = self.scaler.fit_transform(df_copy[columns])
        
        return df_copy
    
    @staticmethod
    def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2,
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets.
        
        Args:
            df: Input dataframe
            target_column: Name of the target column
            test_size: Proportion of test set (0-1)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def get_column_types(df: pd.DataFrame) -> dict:
        """Get column types categorized by numerical and categorical.
        
        Args:
            df: Input dataframe
            
        Returns:
            dict: Dictionary with 'numerical' and 'categorical' column lists
        """
        numerical = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return {
            'numerical': numerical,
            'categorical': categorical
        }
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: List[str], 
                       method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers from numerical columns.
        
        Args:
            df: Input dataframe
            columns: Columns to check for outliers
            method: Method to detect outliers ('iqr' or 'zscore')
            
        Returns:
            pd.DataFrame: Dataframe with outliers removed
        """
        df_copy = df.copy()
        
        for col in columns:
            if col in df_copy.columns:
                if method == 'iqr':
                    Q1 = df_copy[col].quantile(0.25)
                    Q3 = df_copy[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
                elif method == 'zscore':
                    z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                    df_copy = df_copy[z_scores < 3]
        
        return df_copy