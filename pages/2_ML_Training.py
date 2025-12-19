"""
ML Training Page - Train and evaluate machine learning models
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ml_models import MLModelTrainer, XGBOOST_AVAILABLE
from utils.visualizations import Visualizer
from utils.helpers import (
    init_session_state, check_data_loaded, display_metrics,
    validate_data_for_training
)

# Page configuration
st.set_page_config(
    page_title="ML Training - AI Lab Studio",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
init_session_state()

visualizer = Visualizer()

# Title
st.title("🤖 ML Training")
st.markdown("Train and evaluate machine learning models")
st.write("---")

# Check if data is loaded and split
if not check_data_loaded():
    st.stop()

if st.session_state.X_train is None:
    st.warning("⚠️ Please split your data first in the Data Hub page.")
    st.stop()

# Sidebar for model configuration
with st.sidebar:
    st.header("🎯 Model Configuration")
    
    # Task type
    if st.session_state.task_type:
        st.info(f"**Task Type:** {st.session_state.task_type.upper()}")
    else:
        st.session_state.task_type = st.selectbox(
            "Task Type:",
            ["classification", "regression"]
        )
    
    task_type = st.session_state.task_type
    
    # Algorithm selection
    st.subheader("📊 Algorithm")
    
    if task_type == "classification":
        algorithms = ["Logistic Regression", "Random Forest", "SVM"]
        if XGBOOST_AVAILABLE:
            algorithms.append("XGBoost")
    else:
        algorithms = ["Linear Regression", "Random Forest", "SVM"]
        if XGBOOST_AVAILABLE:
            algorithms.append("XGBoost")
    
    algorithm = st.selectbox("Select algorithm:", algorithms)
    
    st.write("---")
    
    # Hyperparameters based on algorithm
    st.subheader("⚙️ Hyperparameters")
    params = {}
    
    if algorithm in ["Logistic Regression", "SVM"]:
        params['C'] = st.slider(
            "C (Regularization)",
            0.01, 10.0, 1.0,
            help="Inverse of regularization strength"
        )
        
        if algorithm == "SVM":
            params['kernel'] = st.selectbox(
                "Kernel",
                ["rbf", "linear", "poly"],
                help="Kernel function for SVM"
            )
    
    if algorithm == "Random Forest":
        params['n_estimators'] = st.slider(
            "Number of Trees",
            10, 500, 100,
            step=10,
            help="Number of trees in the forest"
        )
        params['max_depth'] = st.slider(
            "Max Depth",
            1, 50, 10,
            help="Maximum depth of trees (lower = simpler model)"
        )
        params['min_samples_split'] = st.slider(
            "Min Samples Split",
            2, 20, 2,
            help="Minimum samples required to split a node"
        )
    
    if algorithm == "XGBoost":
        params['n_estimators'] = st.slider(
            "Number of Estimators",
            10, 500, 100,
            step=10
        )
        params['max_depth'] = st.slider(
            "Max Depth",
            1, 20, 6
        )
        params['learning_rate'] = st.slider(
            "Learning Rate",
            0.01, 1.0, 0.1,
            step=0.01
        )

# Main content
tab1, tab2, tab3 = st.tabs([
    "🏋️ Train Model",
    "📊 Model Evaluation",
    "📈 Model Comparison"
])

# Tab 1: Train Model
with tab1:
    st.subheader("Train Your Model")
    
    # Display current configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Algorithm:** {algorithm}")
    with col2:
        st.info(f"**Task:** {task_type.capitalize()}")
    with col3:
        st.info(f"**Features:** {len(st.session_state.feature_columns)}")
    
    st.write("---")
    
    # Show data information
    st.markdown("#### 📊 Training Data Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Samples", st.session_state.X_train.shape[0])
    with col2:
        st.metric("Testing Samples", st.session_state.X_test.shape[0])
    with col3:
        st.metric("Features", st.session_state.X_train.shape[1])
    with col4:
        st.metric("Target", st.session_state.target_column)
    
    st.write("---")
    
    # Feature selection for training
    st.markdown("#### 🔢 Feature Selection")
    
    all_features = st.session_state.X_train.columns.tolist()
    selected_features = st.multiselect(
        "Select features to use for training:",
        all_features,
        default=all_features,
        help="You can train with a subset of features"
    )
    
    if not selected_features:
        st.warning("⚠️ Please select at least one feature!")
        st.stop()
    
    # Model name
    st.markdown("#### 📝 Model Name")
    model_name = st.text_input(
        "Enter a name for this model:",
        value=f"{algorithm.replace(' ', '_')}_{task_type}",
        help="This name will be used when saving the model"
    )
    
    st.write("---")
    
    # Train button
    if st.button("🚀 Train Model", type="primary"):
        try:
            with st.spinner(f"Training {algorithm} model..."):
                # Create trainer
                trainer = MLModelTrainer()
                
                # Create model
                trainer.create_model(algorithm, task_type, params)
                
                # Get selected features
                X_train_selected = st.session_state.X_train[selected_features]
                X_test_selected = st.session_state.X_test[selected_features]
                
                # Train model
                trainer.train_model(
                    X_train_selected,
                    st.session_state.y_train,
                    feature_names=selected_features
                )
                
                # Evaluate model
                if task_type == "classification":
                    metrics = trainer.evaluate_classification(
                        X_test_selected,
                        st.session_state.y_test
                    )
                else:
                    metrics = trainer.evaluate_regression(
                        X_test_selected,
                        st.session_state.y_test
                    )
                
                # Store model and metrics
                st.session_state.current_model = trainer
                st.session_state.current_metrics = metrics
                
                # Add to trained models list
                model_info = {
                    'name': model_name,
                    'algorithm': algorithm,
                    'task_type': task_type,
                    'trainer': trainer,
                    'metrics': metrics,
                    'features': selected_features,
                    'params': params
                }
                st.session_state.trained_models.append(model_info)
                
                st.success(f"✅ Model trained successfully!")
                st.balloons()
                
        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    # Show training history
    if len(st.session_state.trained_models) > 0:
        st.write("---")
        st.markdown("#### 📚 Training History")
        
        history_data = []
        for i, model_info in enumerate(st.session_state.trained_models):
            row = {
                'Model #': i + 1,
                'Name': model_info['name'],
                'Algorithm': model_info['algorithm'],
                'Task': model_info['task_type']
            }
            
            # Add main metric
            if model_info['task_type'] == 'classification':
                row['Accuracy'] = f"{model_info['metrics']['accuracy']:.4f}"
            else:
                row['R²'] = f"{model_info['metrics']['r2']:.4f}"
            
            history_data.append(row)
        
        st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)

# Tab 2: Model Evaluation
with tab2:
    st.subheader("Model Performance Evaluation")
    
    if st.session_state.current_model is None:
        st.info("👈 Train a model first to see evaluation results.")
    else:
        trainer = st.session_state.current_model
        metrics = st.session_state.current_metrics
        
        # Display metrics
        st.markdown("#### 📊 Performance Metrics")
        display_metrics(metrics, task_type)
        
        st.write("---")
        
        # Detailed evaluation based on task type
        if task_type == "classification":
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Confusion Matrix")
                visualizer.plot_confusion_matrix(metrics['confusion_matrix'])
            
            with col2:
                st.markdown("#### 📋 Classification Report")
                st.text(metrics['classification_report'])
            
            # Prediction distribution
            st.write("---")
            st.markdown("#### 📊 Predictions vs Actual")
            
            X_test_selected = st.session_state.X_test[trainer.feature_names]
            y_pred = trainer.predict(X_test_selected)
            
            pred_df = pd.DataFrame({
                'Actual': st.session_state.y_test,
                'Predicted': y_pred
            })
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Actual Distribution**")
                st.bar_chart(pred_df['Actual'].value_counts())
            with col2:
                st.markdown("**Predicted Distribution**")
                st.bar_chart(pred_df['Predicted'].value_counts())
        
        else:  # regression
            st.markdown("#### 📈 Prediction Results")
            
            X_test_selected = st.session_state.X_test[trainer.feature_names]
            y_pred = trainer.predict(X_test_selected)
            
            visualizer.plot_prediction_results(
                st.session_state.y_test.values,
                y_pred,
                'regression'
            )
            
            # Error analysis
            st.write("---")
            st.markdown("#### 📊 Error Analysis")
            
            errors = st.session_state.y_test.values - y_pred
            error_df = pd.DataFrame({
                'Actual': st.session_state.y_test.values,
                'Predicted': y_pred,
                'Error': errors,
                'Absolute Error': np.abs(errors)
            })
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Error Statistics**")
                st.write(f"Mean Error: {errors.mean():.4f}")
                st.write(f"Std Error: {errors.std():.4f}")
                st.write(f"Min Error: {errors.min():.4f}")
                st.write(f"Max Error: {errors.max():.4f}")
            
            with col2:
                st.markdown("**Sample Predictions**")
                st.dataframe(
                    error_df.head(10).round(4),
                    use_container_width=True
                )
        
        # Feature importance
        st.write("---")
        st.markdown("#### ⭐ Feature Importance")
        
        importance_df = trainer.get_feature_importance()
        
        if importance_df is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                visualizer.plot_feature_importance(importance_df, top_n=min(20, len(importance_df)))
            
            with col2:
                st.markdown("**Top 10 Features**")
                st.dataframe(
                    importance_df.head(10).round(4),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info(f"{algorithm} does not provide feature importance scores.")

# Tab 3: Model Comparison
with tab3:
    st.subheader("Compare Multiple Models")
    
    if len(st.session_state.trained_models) == 0:
        st.info("👈 Train multiple models to compare their performance.")
    else:
        st.markdown(f"#### 📊 Comparison of {len(st.session_state.trained_models)} Models")
        
        # Create comparison dataframe
        comparison_data = []
        
        for i, model_info in enumerate(st.session_state.trained_models):
            row = {
                'Model': f"{i+1}. {model_info['name']}",
                'Algorithm': model_info['algorithm'],
                'Features': len(model_info['features'])
            }
            
            # Add metrics
            if model_info['task_type'] == 'classification':
                row['Accuracy'] = model_info['metrics']['accuracy']
                row['Precision'] = model_info['metrics']['precision']
                row['Recall'] = model_info['metrics']['recall']
                row['F1-Score'] = model_info['metrics']['f1_score']
            else:
                row['MAE'] = model_info['metrics']['mae']
                row['RMSE'] = model_info['metrics']['rmse']
                row['R²'] = model_info['metrics']['r2']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.markdown("##### 📋 Comparison Table")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.write("---")
        
        # Visualize comparison
        st.markdown("##### 📊 Performance Visualization")
        
        # Prepare data for visualization
        viz_df = comparison_df.copy()
        
        # Get metric columns (exclude Model, Algorithm, Features)
        metric_cols = [col for col in viz_df.columns 
                      if col not in ['Model', 'Algorithm', 'Features']]
        
        if len(metric_cols) > 0:
            visualizer.plot_metrics_comparison(viz_df[['Model'] + metric_cols])
        
        st.write("---")
        
        # Best model
        st.markdown("##### 🏆 Best Model")
        
        if task_type == 'classification':
            best_idx = comparison_df['Accuracy'].idxmax()
            best_metric = 'Accuracy'
            best_value = comparison_df.loc[best_idx, 'Accuracy']
        else:
            best_idx = comparison_df['R²'].idxmax()
            best_metric = 'R²'
            best_value = comparison_df.loc[best_idx, 'R²']
        
        best_model = comparison_df.loc[best_idx, 'Model']
        best_algo = comparison_df.loc[best_idx, 'Algorithm']
        
        st.success(f"""
        **Best Performing Model:** {best_model}
        
        **Algorithm:** {best_algo}
        
        **{best_metric}:** {best_value:.4f}
        """)
        
        # Model selection for detailed view
        st.write("---")
        st.markdown("##### 🔍 Detailed Model View")
        
        model_names = [m['name'] for m in st.session_state.trained_models]
        selected_model_name = st.selectbox(
            "Select a model to view details:",
            model_names
        )
        
        if selected_model_name:
            selected_model_info = next(
                m for m in st.session_state.trained_models 
                if m['name'] == selected_model_name
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Configuration**")
                st.write(f"Algorithm: {selected_model_info['algorithm']}")
                st.write(f"Task Type: {selected_model_info['task_type']}")
                st.write(f"Features: {len(selected_model_info['features'])}")
                st.write(f"Parameters: {selected_model_info['params']}")
            
            with col2:
                st.markdown("**Performance Metrics**")
                for metric, value in selected_model_info['metrics'].items():
                    if isinstance(value, (int, float)):
                        st.write(f"{metric}: {value:.4f}")
            
            st.markdown("**Features Used**")
            st.write(", ".join(selected_model_info['features']))
