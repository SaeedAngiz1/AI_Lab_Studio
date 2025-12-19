"""
Project Management Page - Save, load, and manage models
"""

import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ml_models import MLModelTrainer, get_model_info
from utils.helpers import (
    init_session_state, get_saved_models, format_datetime
)

# Page configuration
st.set_page_config(
    page_title="Project Management - AI Lab Studio",
    page_icon="💾",
    layout="wide"
)

# Initialize session state
init_session_state()

# Create models directory if it doesn't exist
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Title
st.title("💾 Project Management")
st.markdown("Save, load, and manage your machine learning models")
st.write("---")

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "💾 Save Model",
    "📂 Load Model",
    "📚 Model Registry"
])

# Tab 1: Save Model
with tab1:
    st.subheader("Save Trained Model")
    
    if st.session_state.current_model is None:
        st.info("👈 No model currently trained. Train a model first in the ML Training page.")
    else:
        model = st.session_state.current_model
        
        # Display current model info
        st.markdown("#### 📊 Current Model Information")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Algorithm:** {model.model_type}")
        with col2:
            st.info(f"**Task Type:** {model.task_type}")
        with col3:
            st.info(f"**Features:** {len(model.feature_names)}")
        
        # Display metrics
        if st.session_state.current_metrics:
            st.markdown("#### 📈 Model Performance")
            metrics = st.session_state.current_metrics
            
            if model.task_type == 'classification':
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                with col2:
                    st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                with col3:
                    st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                with col4:
                    st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                with col2:
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                with col3:
                    st.metric("R²", f"{metrics.get('r2', 0):.4f}")
                with col4:
                    st.metric("MSE", f"{metrics.get('mse', 0):.4f}")
        
        st.write("---")
        
        # Model name input
        st.markdown("#### 📝 Save Configuration")
        
        default_name = f"{model.model_type.replace(' ', '_')}_{model.task_type}"
        model_name = st.text_input(
            "Model Name:",
            value=default_name,
            help="Enter a descriptive name for your model"
        )
        
        model_description = st.text_area(
            "Description (optional):",
            placeholder="Enter a description of this model...",
            help="Add notes about the model, its purpose, or any other relevant information"
        )
        
        # Save button
        if st.button("💾 Save Model", type="primary"):
            if not model_name:
                st.error("❌ Please enter a model name!")
            else:
                try:
                    with st.spinner("Saving model..."):
                        # Prepare metrics for saving
                        save_metrics = st.session_state.current_metrics.copy() if st.session_state.current_metrics else {}
                        
                        # Convert numpy arrays to lists for JSON serialization
                        for key, value in save_metrics.items():
                            if hasattr(value, 'tolist'):
                                save_metrics[key] = value.tolist()
                            elif not isinstance(value, (int, float, str, bool, list, dict, type(None))):
                                save_metrics[key] = str(value)
                        
                        # Add description to metrics
                        if model_description:
                            save_metrics['description'] = model_description
                        
                        # Save model
                        filepath = model.save_model(
                            model_name=model_name,
                            models_dir=MODELS_DIR,
                            metrics=save_metrics
                        )
                        
                        st.success(f"✅ Model saved successfully!")
                        st.info(f"📁 Saved to: {filepath}")
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"❌ Error saving model: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
        
        st.write("---")
        
        # Feature information
        with st.expander("📋 View Model Features"):
            st.write("**Features used by this model:**")
            for i, feature in enumerate(model.feature_names, 1):
                st.write(f"{i}. {feature}")

# Tab 2: Load Model
with tab2:
    st.subheader("Load Saved Model")
    
    # Get list of saved models
    saved_models = get_saved_models(MODELS_DIR)
    
    if not saved_models:
        st.info("📂 No saved models found. Save a model first in the 'Save Model' tab.")
    else:
        st.markdown(f"#### 📚 Available Models ({len(saved_models)})")
        
        # Select model to load
        selected_model_file = st.selectbox(
            "Select a model to load:",
            saved_models,
            format_func=lambda x: x.replace('.pkl', ''),
            help="Choose a saved model to load into memory"
        )
        
        if selected_model_file:
            model_path = os.path.join(MODELS_DIR, selected_model_file)
            
            # Show model information
            try:
                model_info = get_model_info(model_path)
                
                st.write("---")
                st.markdown("#### 📊 Model Information")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**Algorithm:** {model_info['model_type']}")
                with col2:
                    st.info(f"**Task Type:** {model_info['task_type']}")
                with col3:
                    st.info(f"**Features:** {len(model_info['feature_names'])}")
                
                # Show metrics
                if model_info['metrics']:
                    st.markdown("#### 📈 Performance Metrics")
                    
                    metrics = model_info['metrics']
                    
                    # Show description if available
                    if 'description' in metrics:
                        st.markdown(f"**Description:** {metrics['description']}")
                        st.write("---")
                    
                    if model_info['task_type'] == 'classification':
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                        with col2:
                            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                        with col3:
                            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                        with col4:
                            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
                    else:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                        with col2:
                            st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                        with col3:
                            st.metric("R²", f"{metrics.get('r2', 0):.4f}")
                        with col4:
                            st.metric("MSE", f"{metrics.get('mse', 0):.4f}")
                
                # Show features
                with st.expander("📋 View Model Features"):
                    st.write("**Features used by this model:**")
                    for i, feature in enumerate(model_info['feature_names'], 1):
                        st.write(f"{i}. {feature}")
                
                st.write("---")
                
                # Load button
                if st.button("📂 Load This Model", type="primary"):
                    try:
                        with st.spinner("Loading model..."):
                            # Create new trainer instance
                            trainer = MLModelTrainer()
                            
                            # Load the model
                            metrics = trainer.load_model(model_path)
                            
                            # Update session state
                            st.session_state.current_model = trainer
                            st.session_state.current_metrics = metrics
                            
                            st.success("✅ Model loaded successfully!")
                            st.info("You can now use this model in the Prediction page.")
                            st.balloons()
                            
                    except Exception as e:
                        st.error(f"❌ Error loading model: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
                
            except Exception as e:
                st.error(f"❌ Error reading model info: {str(e)}")

# Tab 3: Model Registry
with tab3:
    st.subheader("Model Registry")
    st.markdown("View and manage all saved models")
    
    # Get list of saved models
    saved_models = get_saved_models(MODELS_DIR)
    
    if not saved_models:
        st.info("📂 No saved models found. Train and save models to see them here.")
    else:
        st.markdown(f"#### 📚 Saved Models ({len(saved_models)})")
        
        # Create a table with model information
        model_data = []
        
        for model_file in saved_models:
            model_path = os.path.join(MODELS_DIR, model_file)
            
            try:
                info = get_model_info(model_path)
                
                # Extract timestamp from filename
                timestamp = model_file.split('_')[-1].replace('.pkl', '')
                formatted_date = format_datetime(timestamp)
                
                # Get file size
                file_size = os.path.getsize(model_path) / 1024  # KB
                
                row = {
                    'Model Name': model_file.replace('.pkl', ''),
                    'Algorithm': info['model_type'],
                    'Task Type': info['task_type'].capitalize(),
                    'Features': len(info['feature_names']),
                    'Date Created': formatted_date,
                    'Size (KB)': f"{file_size:.2f}"
                }
                
                # Add main performance metric
                if info['metrics']:
                    if info['task_type'] == 'classification':
                        row['Accuracy'] = f"{info['metrics'].get('accuracy', 0):.4f}"
                    else:
                        row['R²'] = f"{info['metrics'].get('r2', 0):.4f}"
                
                model_data.append(row)
                
            except Exception as e:
                st.warning(f"⚠️ Could not read info for {model_file}: {str(e)}")
        
        if model_data:
            # Display as dataframe
            df = pd.DataFrame(model_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.write("---")
            
            # Model management
            st.markdown("#### 🔧 Model Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 📊 Export Model Information")
                
                # Export all model info as JSON
                if st.button("📥 Export Registry as JSON"):
                    try:
                        export_data = {
                            'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'total_models': len(model_data),
                            'models': model_data
                        }
                        
                        json_str = json.dumps(export_data, indent=2)
                        
                        st.download_button(
                            label="⬇️ Download JSON",
                            data=json_str,
                            file_name="model_registry.json",
                            mime="application/json"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error exporting: {str(e)}")
                
                # Export as CSV
                if st.button("📥 Export Registry as CSV"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv,
                        file_name="model_registry.csv",
                        mime="text/csv"
                    )
            
            with col2:
                st.markdown("##### 🗑️ Delete Model")
                
                model_to_delete = st.selectbox(
                    "Select model to delete:",
                    saved_models,
                    format_func=lambda x: x.replace('.pkl', ''),
                    key="delete_select"
                )
                
                if model_to_delete:
                    if st.button("🗑️ Delete Selected Model", type="secondary"):
                        # Confirmation
                        st.warning(f"⚠️ Are you sure you want to delete '{model_to_delete}'?")
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            if st.button("✅ Yes, Delete", key="confirm_delete"):
                                try:
                                    model_path = os.path.join(MODELS_DIR, model_to_delete)
                                    os.remove(model_path)
                                    st.success(f"✅ Model deleted: {model_to_delete}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error deleting model: {str(e)}")
                        
                        with col_b:
                            if st.button("❌ Cancel", key="cancel_delete"):
                                st.info("Deletion cancelled.")
            
            st.write("---")
            
            # Statistics
            st.markdown("#### 📈 Registry Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Models", len(model_data))
            
            with col2:
                classification_count = sum(1 for m in model_data if m['Task Type'] == 'Classification')
                st.metric("Classification Models", classification_count)
            
            with col3:
                regression_count = sum(1 for m in model_data if m['Task Type'] == 'Regression')
                st.metric("Regression Models", regression_count)
            
            with col4:
                total_size = sum(float(m['Size (KB)']) for m in model_data)
                st.metric("Total Size", f"{total_size:.2f} KB")
            
            # Algorithm distribution
            st.write("---")
            st.markdown("#### 📊 Algorithm Distribution")
            
            algo_counts = pd.Series([m['Algorithm'] for m in model_data]).value_counts()
            st.bar_chart(algo_counts)
        
        st.write("---")
        
        # Bulk operations
        with st.expander("🔧 Bulk Operations"):
            st.markdown("##### ⚠️ Danger Zone")
            
            st.warning("These operations cannot be undone!")
            
            if st.button("🗑️ Delete All Models", type="secondary"):
                st.error("⚠️ This will delete ALL saved models. Are you absolutely sure?")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if st.button("✅ Yes, Delete All", key="confirm_delete_all"):
                        try:
                            deleted_count = 0
                            for model_file in saved_models:
                                model_path = os.path.join(MODELS_DIR, model_file)
                                os.remove(model_path)
                                deleted_count += 1
                            
                            st.success(f"✅ Deleted {deleted_count} models")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                
                with col_b:
                    if st.button("❌ Cancel", key="cancel_delete_all"):
                        st.info("Operation cancelled.")

# Footer with tips
st.write("---")
st.markdown("### 💡 Project Management Tips")

with st.expander("📚 Learn More"):
    st.markdown("""
    #### Saving Models:
    - Give your models descriptive names for easy identification
    - Add descriptions to remember model purposes and configurations
    - Models are saved with timestamps automatically
    
    #### Loading Models:
    - Review model information before loading
    - Check that the model's features match your prediction data
    - Load a model to use it in the Prediction page
    
    #### Model Registry:
    - View all saved models in one place
    - Export registry for documentation or reporting
    - Delete old or unused models to save space
    - Compare different models' performance metrics
    
    #### Best Practices:
    - Regularly backup your models directory
    - Use clear naming conventions
    - Document model purposes and use cases
    - Clean up old models periodically
    """)
