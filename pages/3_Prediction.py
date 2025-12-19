"""
Prediction Page - Make predictions with trained models
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import (
    init_session_state, check_model_trained, create_download_link
)

# Page configuration
st.set_page_config(
    page_title="Prediction - AI Lab Studio",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
init_session_state()

# Title
st.title("🎯 Prediction")
st.markdown("Make predictions with your trained models")
st.write("---")

# Check if model is trained
if not check_model_trained():
    st.stop()

# Get current model
model = st.session_state.current_model
model_name = model.model_type
task_type = model.task_type
feature_names = model.feature_names

# Display model information
st.markdown("### 📊 Current Model Information")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info(f"**Model:** {model_name}")
with col2:
    st.info(f"**Task:** {task_type.capitalize()}")
with col3:
    st.info(f"**Features:** {len(feature_names)}")
with col4:
    if st.session_state.current_metrics:
        if task_type == 'classification':
            metric_value = st.session_state.current_metrics.get('accuracy', 0)
            st.info(f"**Accuracy:** {metric_value:.4f}")
        else:
            metric_value = st.session_state.current_metrics.get('r2', 0)
            st.info(f"**R²:** {metric_value:.4f}")

st.write("---")

# Create tabs for different prediction modes
tab1, tab2 = st.tabs(["🔢 Single Prediction", "📊 Batch Prediction"])

# Tab 1: Single Prediction
with tab1:
    st.subheader("Single Prediction")
    st.markdown("Enter values for each feature to get a prediction.")
    
    # Create input form
    st.markdown("#### 📝 Input Features")
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    input_values = {}
    
    # Split features into two columns
    mid_point = (len(feature_names) + 1) // 2
    
    with col1:
        for feature in feature_names[:mid_point]:
            # Get feature statistics from training data if available
            if st.session_state.X_train is not None and feature in st.session_state.X_train.columns:
                feature_data = st.session_state.X_train[feature]
                min_val = float(feature_data.min())
                max_val = float(feature_data.max())
                mean_val = float(feature_data.mean())
                
                input_values[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    help=f"Range: {min_val:.2f} to {max_val:.2f}"
                )
            else:
                input_values[feature] = st.number_input(
                    f"{feature}",
                    value=0.0
                )
    
    with col2:
        for feature in feature_names[mid_point:]:
            # Get feature statistics from training data if available
            if st.session_state.X_train is not None and feature in st.session_state.X_train.columns:
                feature_data = st.session_state.X_train[feature]
                min_val = float(feature_data.min())
                max_val = float(feature_data.max())
                mean_val = float(feature_data.mean())
                
                input_values[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    help=f"Range: {min_val:.2f} to {max_val:.2f}"
                )
            else:
                input_values[feature] = st.number_input(
                    f"{feature}",
                    value=0.0
                )
    
    st.write("---")
    
    # Predict button
    if st.button("🎯 Make Prediction", type="primary"):
        try:
            # Create dataframe from inputs
            input_df = pd.DataFrame([input_values])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Display prediction
            st.markdown("### 🎊 Prediction Result")
            
            if task_type == 'classification':
                # Get prediction probabilities if available
                try:
                    probabilities = model.predict_proba(input_df)[0]
                    
                    # Display prediction
                    st.success(f"**Predicted Class:** {prediction}")
                    
                    # Display probabilities
                    st.markdown("#### 📊 Prediction Confidence")
                    
                    # Get unique classes from training data
                    if st.session_state.y_train is not None:
                        classes = sorted(st.session_state.y_train.unique())
                        
                        prob_df = pd.DataFrame({
                            'Class': classes,
                            'Probability': probabilities,
                            'Percentage': [f"{p*100:.2f}%" for p in probabilities]
                        })
                        
                        # Display as table
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)
                        
                        # Display as bar chart
                        st.bar_chart(prob_df.set_index('Class')['Probability'])
                    else:
                        # Just show probabilities
                        for i, prob in enumerate(probabilities):
                            st.write(f"Class {i}: {prob:.4f} ({prob*100:.2f}%)")
                    
                except:
                    # Model doesn't support probability predictions
                    st.success(f"**Predicted Class:** {prediction}")
            
            else:  # regression
                st.success(f"**Predicted Value:** {prediction:.4f}")
                
                # Show confidence interval (if using test data statistics)
                if st.session_state.y_test is not None and st.session_state.current_metrics:
                    rmse = st.session_state.current_metrics.get('rmse', 0)
                    
                    st.markdown("#### 📊 Prediction Range (±1 RMSE)")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Lower Bound", f"{prediction - rmse:.4f}")
                    with col2:
                        st.metric("Prediction", f"{prediction:.4f}")
                    with col3:
                        st.metric("Upper Bound", f"{prediction + rmse:.4f}")
            
            st.write("---")
            
            # Display input values used
            with st.expander("📋 View Input Values"):
                st.dataframe(input_df.T, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

# Tab 2: Batch Prediction
with tab2:
    st.subheader("Batch Prediction")
    st.markdown("Upload a CSV file to make predictions for multiple samples at once.")
    
    # Instructions
    st.info(f"""
    **Required columns in your CSV file:**
    
    {', '.join(feature_names)}
    
    The file should contain these exact column names with data for prediction.
    """)
    
    # Show sample format
    with st.expander("📋 View Sample Format"):
        sample_df = pd.DataFrame(columns=feature_names)
        
        # Add sample rows if training data exists
        if st.session_state.X_train is not None:
            # Take first 3 rows from training data as example
            sample_df = st.session_state.X_train[feature_names].head(3)
        
        st.dataframe(sample_df, use_container_width=True)
        
        # Provide download link for sample
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Sample Template",
            data=csv,
            file_name="prediction_template.csv",
            mime="text/csv"
        )
    
    st.write("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch prediction",
        type=['csv'],
        help="Upload a CSV file with the required feature columns"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            batch_data = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded: {batch_data.shape[0]} rows")
            
            # Display preview
            st.markdown("#### 📊 Data Preview")
            st.dataframe(batch_data.head(10), use_container_width=True)
            
            st.write("---")
            
            # Validate columns
            missing_cols = [col for col in feature_names if col not in batch_data.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                # Select only required features
                batch_features = batch_data[feature_names]
                
                st.markdown("#### 🎯 Make Batch Predictions")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Samples to Predict", len(batch_features))
                with col2:
                    st.metric("Features", len(feature_names))
                with col3:
                    st.metric("Model", model_name)
                
                if st.button("🚀 Run Batch Prediction", type="primary"):
                    try:
                        with st.spinner("Making predictions..."):
                            # Make predictions
                            predictions = model.predict(batch_features)
                            
                            # Create results dataframe
                            results_df = batch_data.copy()
                            results_df['Prediction'] = predictions
                            
                            # Add probabilities for classification
                            if task_type == 'classification':
                                try:
                                    probabilities = model.predict_proba(batch_features)
                                    
                                    # Add probability columns
                                    classes = sorted(st.session_state.y_train.unique())
                                    for i, cls in enumerate(classes):
                                        results_df[f'Probability_Class_{cls}'] = probabilities[:, i]
                                    
                                except:
                                    pass  # Model doesn't support probabilities
                            
                            st.success("✅ Predictions completed!")
                            
                            # Display results
                            st.markdown("#### 📊 Prediction Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Summary statistics
                            st.write("---")
                            st.markdown("#### 📈 Prediction Summary")
                            
                            if task_type == 'classification':
                                # Count predictions per class
                                pred_counts = pd.Series(predictions).value_counts().sort_index()
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Prediction Distribution**")
                                    st.dataframe(
                                        pred_counts.reset_index(name='Count').rename(columns={'index': 'Class'}),
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                
                                with col2:
                                    st.markdown("**Distribution Chart**")
                                    st.bar_chart(pred_counts)
                            
                            else:  # regression
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Mean Prediction", f"{predictions.mean():.4f}")
                                with col2:
                                    st.metric("Std Dev", f"{predictions.std():.4f}")
                                with col3:
                                    st.metric("Min Prediction", f"{predictions.min():.4f}")
                                with col4:
                                    st.metric("Max Prediction", f"{predictions.max():.4f}")
                                
                                # Histogram of predictions
                                st.markdown("**Prediction Distribution**")
                                st.bar_chart(pd.Series(predictions).value_counts().sort_index())
                            
                            st.write("---")
                            
                            # Download results
                            st.markdown("#### 💾 Download Results")
                            
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="⬇️ Download Predictions as CSV",
                                data=csv,
                                file_name="predictions_results.csv",
                                mime="text/csv"
                            )
                            
                    except Exception as e:
                        st.error(f"❌ Error making predictions: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

# Additional features
st.write("---")
st.markdown("### 💡 Tips for Better Predictions")

with st.expander("📚 Learn More"):
    st.markdown("""
    #### Single Prediction Tips:
    - Enter realistic values within the feature ranges shown
    - Check the training data ranges to understand valid inputs
    - For classification, higher confidence (probability) indicates more certain predictions
    
    #### Batch Prediction Tips:
    - Ensure your CSV has all required feature columns
    - Column names must match exactly (case-sensitive)
    - Remove any extra columns that aren't features
    - Check for missing values before uploading
    
    #### Understanding Results:
    - **Classification**: Predicted class and probability for each class
    - **Regression**: Predicted numerical value with confidence range
    - Download results for further analysis or reporting
    """)
