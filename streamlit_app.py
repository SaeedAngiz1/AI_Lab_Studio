import streamlit as st
from utils.helpers import init_session_state

# Page configuration
st.set_page_config(
    page_title="AI Lab Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
init_session_state()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #3b82f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Main page
st.markdown('<h1 class="main-header">🧪 AI Lab Studio</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your Complete Machine Learning Development Platform</p>', unsafe_allow_html=True)

# Welcome message
st.write("---")
st.markdown("""
Welcome to **AI Lab Studio**! 🚀 This comprehensive platform provides everything you need for 
end-to-end machine learning development, from data preparation to model deployment.

### 🎯 Quick Start Guide

Navigate through the pages using the sidebar to access different features:
""")

# Feature overview
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-box">
        <h3>📊 Data Hub</h3>
        <p>Upload, explore, and preprocess your datasets with powerful tools:</p>
        <ul>
            <li>CSV file upload</li>
            <li>Data preview and statistics</li>
            <li>Handle missing values</li>
            <li>Encode categorical variables</li>
            <li>Feature scaling</li>
            <li>Train/test split</li>
            <li>Interactive visualizations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
        <h3>🎯 Prediction</h3>
        <p>Make predictions with your trained models:</p>
        <ul>
            <li>Single prediction interface</li>
            <li>Batch predictions from CSV</li>
            <li>Prediction confidence scores</li>
            <li>Export predictions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <h3>🤖 ML Training</h3>
        <p>Train and evaluate machine learning models:</p>
        <ul>
            <li>Multiple algorithms (Logistic Regression, Random Forest, XGBoost, SVM)</li>
            <li>Hyperparameter tuning</li>
            <li>Comprehensive evaluation metrics</li>
            <li>Feature importance analysis</li>
            <li>Model comparison</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
        <h3>💾 Project Management</h3>
        <p>Manage your machine learning projects:</p>
        <ul>
            <li>Save trained models</li>
            <li>Load existing models</li>
            <li>Model registry with metadata</li>
            <li>Export model information</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# Sample datasets information
st.markdown("### 📁 Sample Datasets")
st.info("""
**Get started immediately with our included sample datasets:**

1. **Iris Dataset** (Classification) - `sample_data/iris.csv`
   - 150 samples, 4 features
   - Predict flower species (setosa, versicolor, virginica)

2. **California Housing Dataset** (Regression) - `sample_data/boston_housing.csv`
   - 20,640 samples, 8 features
   - Predict median house prices

Navigate to the **Data Hub** page to load these datasets and start exploring!
""")

# Current session status
st.write("---")
st.markdown("### 📈 Current Session Status")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    if st.session_state.data is not None:
        st.success(f"✅ Data Loaded: {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} columns")
    else:
        st.info("📂 No data loaded yet")

with status_col2:
    if st.session_state.current_model is not None:
        st.success(f"✅ Model Trained: {st.session_state.current_model.model_type}")
    else:
        st.info("🤖 No model trained yet")

with status_col3:
    trained_count = len(st.session_state.trained_models)
    if trained_count > 0:
        st.success(f"✅ {trained_count} model(s) in session")
    else:
        st.info("📊 No models in session")

# Footer
st.write("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>AI Lab Studio</strong> - Empowering Machine Learning Development</p>
    <p>Built with Streamlit 🎈 | Powered by scikit-learn, XGBoost, and more</p>
</div>
""", unsafe_allow_html=True)