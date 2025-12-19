"""Visualization utilities for AI Lab Studio."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import List, Optional


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


class Visualizer:
    """Handles all visualization tasks."""
    
    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, figsize: tuple = (12, 8)):
        """Plot correlation heatmap for numerical features.
        
        Args:
            df: Input dataframe
            figsize: Figure size
        """
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(numerical_cols) == 0:
            st.warning("No numerical columns to plot correlation.")
            return
        
        corr_matrix = df[numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    @staticmethod
    def plot_distribution(df: pd.DataFrame, column: str, bins: int = 30):
        """Plot distribution for a numerical column.
        
        Args:
            df: Input dataframe
            column: Column name
            bins: Number of bins for histogram
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(df[column].dropna(), bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel(column, fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(df[column].dropna(), vert=True)
        ax2.set_ylabel(column, fontsize=12)
        ax2.set_title(f'Box Plot of {column}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    @staticmethod
    def plot_categorical_distribution(df: pd.DataFrame, column: str, top_n: int = 20):
        """Plot distribution for a categorical column.
        
        Args:
            df: Input dataframe
            column: Column name
            top_n: Show top N categories
        """
        value_counts = df[column].value_counts().head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Distribution of {column} (Top {top_n})', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, labels: Optional[List] = None):
        """Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            labels: Class labels
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, ax=ax,
                   cbar_kws={'shrink': 0.8})
        
        if labels:
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20):
        """Plot feature importance.
        
        Args:
            importance_df: Dataframe with 'feature' and 'importance' columns
            top_n: Number of top features to show
        """
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(top_features)), top_features['importance'], color='forestgreen')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    @staticmethod
    def plot_metrics_comparison(metrics_df: pd.DataFrame):
        """Plot comparison of multiple models.
        
        Args:
            metrics_df: Dataframe with model names and metrics
        """
        if len(metrics_df) == 0:
            st.warning("No metrics to compare.")
            return
        
        # Get metric columns (exclude model name)
        metric_cols = [col for col in metrics_df.columns if col != 'Model']
        
        fig = go.Figure()
        
        for metric in metric_cols:
            fig.add_trace(go.Bar(
                name=metric,
                x=metrics_df['Model'],
                y=metrics_df[metric],
                text=metrics_df[metric].round(3),
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def plot_scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None):
        """Plot scatter plot.
        
        Args:
            df: Input dataframe
            x: X-axis column
            y: Y-axis column
            color: Column for color coding
        """
        fig = px.scatter(df, x=x, y=y, color=color, 
                        title=f'{y} vs {x}',
                        template='plotly_white',
                        height=500)
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def plot_line(df: pd.DataFrame, x: str, y: str):
        """Plot line chart.
        
        Args:
            df: Input dataframe
            x: X-axis column
            y: Y-axis column
        """
        fig = px.line(df, x=x, y=y,
                     title=f'{y} over {x}',
                     template='plotly_white',
                     height=500)
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def plot_prediction_results(y_true: np.ndarray, y_pred: np.ndarray, task_type: str):
        """Plot prediction results comparison.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            task_type: 'classification' or 'regression'
        """
        if task_type == 'regression':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Scatter plot
            ax1.scatter(y_true, y_pred, alpha=0.6, color='blue')
            ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', lw=2, label='Perfect Prediction')
            ax1.set_xlabel('True Values', fontsize=12)
            ax1.set_ylabel('Predicted Values', fontsize=12)
            ax1.set_title('Predicted vs True Values', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Residuals plot
            residuals = y_true - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.6, color='green')
            ax2.axhline(y=0, color='r', linestyle='--', lw=2)
            ax2.set_xlabel('Predicted Values', fontsize=12)
            ax2.set_ylabel('Residuals', fontsize=12)
            ax2.set_title('Residuals Plot', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    @staticmethod
    def plot_data_overview(df: pd.DataFrame):
        """Plot overview of dataset.
        
        Args:
            df: Input dataframe
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Missing Values', 'Data Types', 'Column Count', 'Row Statistics'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}],
                  [{'type': 'indicator'}, {'type': 'indicator'}]]
        )
        
        # Missing values
        missing = df.isnull().sum().sort_values(ascending=False).head(10)
        if missing.sum() > 0:
            fig.add_trace(
                go.Bar(x=missing.index, y=missing.values, name='Missing Values'),
                row=1, col=1
            )
        
        # Data types
        dtype_counts = df.dtypes.value_counts()
        fig.add_trace(
            go.Pie(labels=dtype_counts.index.astype(str), values=dtype_counts.values, name='Data Types'),
            row=1, col=2
        )
        
        # Column count
        fig.add_trace(
            go.Indicator(
                mode='number',
                value=len(df.columns),
                title={'text': 'Total Columns'},
            ),
            row=2, col=1
        )
        
        # Row count
        fig.add_trace(
            go.Indicator(
                mode='number',
                value=len(df),
                title={'text': 'Total Rows'},
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)