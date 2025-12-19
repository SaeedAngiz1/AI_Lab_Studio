"""Helper utilities for AI Lab Studio."""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import json


def init_session_state():
    """Initialize session state variables."""
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = []
    
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    
    if 'feature_columns' not in st.session_state:
        st.session_state.feature_columns = []
    
    if 'task_type' not in st.session_state:
        st.session_state.task_type = None
    
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = None


def display_dataframe(df: pd.DataFrame, title: str = "Data Preview", 
                     max_rows: int = 100):
    """Display dataframe with pagination.
    
    Args:
        df: Dataframe to display
        title: Title for the dataframe
        max_rows: Maximum rows per page
    """
    st.subheader(title)
    
    if df is None or len(df) == 0:
        st.info("No data to display.")
        return
    
    # Pagination
    total_rows = len(df)
    total_pages = (total_rows - 1) // max_rows + 1
    
    if total_pages > 1:
        page = st.slider("Page", 1, total_pages, 1, key=f"page_{title}")
        start_idx = (page - 1) * max_rows
        end_idx = min(start_idx + max_rows, total_rows)
        st.write(f"Showing rows {start_idx + 1} to {end_idx} of {total_rows}")
        st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)


def display_metrics(metrics: Dict[str, Any], task_type: str):
    """Display model metrics in a nice format.
    
    Args:
        metrics: Dictionary of metrics
        task_type: 'classification' or 'regression'
    """
    if task_type == 'classification':
        cols = st.columns(4)
        
        with cols[0]:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        with cols[1]:
            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        with cols[2]:
            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        with cols[3]:
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
    
    elif task_type == 'regression':
        cols = st.columns(4)
        
        with cols[0]:
            st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
        with cols[1]:
            st.metric("MSE", f"{metrics.get('mse', 0):.4f}")
        with cols[2]:
            st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
        with cols[3]:
            st.metric("R²", f"{metrics.get('r2', 0):.4f}")


def check_data_loaded(show_warning: bool = True) -> bool:
    """Check if data is loaded in session state.
    
    Args:
        show_warning: Whether to show warning message
        
    Returns:
        bool: True if data is loaded, False otherwise
    """
    if st.session_state.data is None:
        if show_warning:
            st.warning("⚠️ Please load data first in the Data Hub page.")
        return False
    return True


def check_model_trained(show_warning: bool = True) -> bool:
    """Check if a model is trained.
    
    Args:
        show_warning: Whether to show warning message
        
    Returns:
        bool: True if model is trained, False otherwise
    """
    if st.session_state.current_model is None:
        if show_warning:
            st.warning("⚠️ Please train a model first in the ML Training page.")
        return False
    return True


def get_saved_models(models_dir: str = 'models') -> List[str]:
    """Get list of saved model files.
    
    Args:
        models_dir: Directory containing saved models
        
    Returns:
        List of model filenames
    """
    if not os.path.exists(models_dir):
        return []
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    return sorted(model_files, reverse=True)


def format_datetime(timestamp: str) -> str:
    """Format timestamp string.
    
    Args:
        timestamp: Timestamp string in format YYYYMMDD_HHMMSS
        
    Returns:
        Formatted datetime string
    """
    try:
        dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp


def create_download_link(df: pd.DataFrame, filename: str, link_text: str):
    """Create a download link for dataframe.
    
    Args:
        df: Dataframe to download
        filename: Name for the downloaded file
        link_text: Text to display for the download button
    """
    csv = df.to_csv(index=False)
    st.download_button(
        label=link_text,
        data=csv,
        file_name=filename,
        mime='text/csv'
    )


def show_info_box(title: str, content: str, type: str = "info"):
    """Show an information box.
    
    Args:
        title: Title of the info box
        content: Content to display
        type: Type of message ('info', 'success', 'warning', 'error')
    """
    if type == "info":
        st.info(f"**{title}**\n\n{content}")
    elif type == "success":
        st.success(f"**{title}**\n\n{content}")
    elif type == "warning":
        st.warning(f"**{title}**\n\n{content}")
    elif type == "error":
        st.error(f"**{title}**\n\n{content}")


def detect_task_type(y: pd.Series) -> str:
    """Automatically detect if task is classification or regression.
    
    Args:
        y: Target variable
        
    Returns:
        str: 'classification' or 'regression'
    """
    # If target is object/string type, it's classification
    if y.dtype == 'object' or y.dtype.name == 'category':
        return 'classification'
    
    # If target is numeric, check unique values
    unique_count = y.nunique()
    total_count = len(y)
    
    # If unique values are less than 10% of total or less than 20, it's likely classification
    if unique_count < 20 or (unique_count / total_count) < 0.05:
        return 'classification'
    else:
        return 'regression'


def validate_data_for_training(data: pd.DataFrame, target_column: str) -> tuple:
    """Validate if data is ready for training.
    
    Args:
        data: Input dataframe
        target_column: Name of target column
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if data is None:
        return False, "No data loaded."
    
    if target_column not in data.columns:
        return False, f"Target column '{target_column}' not found in data."
    
    if len(data.columns) < 2:
        return False, "Data must have at least 2 columns (features and target)."
    
    if data.isnull().any().any():
        return False, "Data contains missing values. Please handle them in Data Hub."
    
    # Check if there are any object columns (besides target if it's classification)
    feature_cols = [col for col in data.columns if col != target_column]
    object_cols = data[feature_cols].select_dtypes(include=['object']).columns
    
    if len(object_cols) > 0:
        return False, f"Data contains categorical columns: {list(object_cols)}. Please encode them in Data Hub."
    
    return True, ""


def display_data_stats(df: pd.DataFrame):
    """Display basic statistics about the dataframe.
    
    Args:
        df: Input dataframe
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    with col4:
        st.metric("Duplicates", f"{df.duplicated().sum():,}")