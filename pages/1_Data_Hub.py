"""
Data Hub Page - Upload, explore, and preprocess data
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import DataProcessor
from utils.visualizations import Visualizer
from utils.helpers import (
    init_session_state, display_dataframe, display_data_stats,
    show_info_box, detect_task_type
)

# Page configuration
st.set_page_config(
    page_title="Data Hub - AI Lab Studio",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
init_session_state()

# Initialize processor
if st.session_state.data_processor is None:
    st.session_state.data_processor = DataProcessor()

processor = st.session_state.data_processor
visualizer = Visualizer()

# Title
st.title("📊 Data Hub")
st.markdown("Upload, explore, and preprocess your datasets")
st.write("---")

# Sidebar for data loading
with st.sidebar:
    st.header("📁 Data Loading")
    
    # Option to load sample data or upload
    data_source = st.radio(
        "Choose data source:",
        ["Upload CSV", "Load Sample Data"]
    )
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="Upload a CSV file to get started"
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Loading data..."):
                    df = processor.load_data(uploaded_file)
                    st.session_state.data = df
                    st.session_state.processed_data = df.copy()
                    st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    else:  # Load Sample Data
        sample_choice = st.selectbox(
            "Choose sample dataset:",
            ["Select...", "Iris (Classification)", "California Housing (Regression)"]
        )
        
        if sample_choice != "Select...":
            if st.button("Load Sample Dataset"):
                try:
                    if sample_choice == "Iris (Classification)":
                        df = pd.read_csv("sample_data/iris.csv")
                    else:
                        df = pd.read_csv("sample_data/boston_housing.csv")
                    
                    st.session_state.data = df
                    st.session_state.processed_data = df.copy()
                    st.success(f"✅ {sample_choice} loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading sample data: {str(e)}")
    
    st.write("---")
    
    # Data actions
    if st.session_state.data is not None:
        st.header("🔧 Data Actions")
        
        if st.button("🔄 Reset to Original"):
            st.session_state.processed_data = st.session_state.data.copy()
            st.success("Data reset to original!")
            st.rerun()
        
        if st.button("💾 Save Processed Data"):
            st.session_state.data = st.session_state.processed_data.copy()
            st.success("Processed data saved!")

# Main content
if st.session_state.data is None:
    st.info("👈 Please load data using the sidebar to get started.")
    
    # Show instructions
    st.markdown("""
    ### Getting Started
    
    1. **Upload your own CSV file** or **load a sample dataset** from the sidebar
    2. Explore the data with preview, statistics, and visualizations
    3. Preprocess the data (handle missing values, encode categories, scale features)
    4. Split data into training and test sets
    5. Move to ML Training page to build models
    
    ### Sample Datasets Available
    - **Iris Dataset**: Perfect for classification tasks
    - **California Housing Dataset**: Great for regression problems
    """)
else:
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data Overview",
        "🔧 Preprocessing",
        "📊 Visualizations",
        "✂️ Train/Test Split"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.subheader("Dataset Overview")
        
        # Display basic statistics
        display_data_stats(st.session_state.processed_data)
        
        st.write("---")
        
        # Data preview
        display_dataframe(st.session_state.processed_data, "Data Preview", max_rows=50)
        
        st.write("---")
        
        # Detailed information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Summary Statistics")
            st.dataframe(
                st.session_state.processed_data.describe(),
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 🔍 Data Info")
            
            info = processor.get_data_info(st.session_state.processed_data)
            
            # Display data types
            st.markdown("**Data Types:**")
            dtypes_df = pd.DataFrame({
                'Column': list(info['dtypes'].keys()),
                'Type': [str(v) for v in info['dtypes'].values()]
            })
            st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
            
            st.markdown(f"**Memory Usage:** {info['memory_usage']:.2f} MB")
            st.markdown(f"**Duplicate Rows:** {info['duplicates']}")
        
        st.write("---")
        
        # Missing values analysis
        st.markdown("#### 🔍 Missing Values Analysis")
        missing_df = pd.DataFrame({
            'Column': list(info['missing_values'].keys()),
            'Missing Count': list(info['missing_values'].values()),
            'Missing %': [f"{v:.2f}%" for v in info['missing_percentage'].values()]
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No missing values found!")
    
    # Tab 2: Preprocessing
    with tab2:
        st.subheader("Data Preprocessing Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔧 Handle Missing Values")
            
            # Check if there are missing values
            missing_cols = [col for col in st.session_state.processed_data.columns 
                          if st.session_state.processed_data[col].isnull().sum() > 0]
            
            if len(missing_cols) > 0:
                strategy = st.selectbox(
                    "Strategy:",
                    ["drop", "mean", "median", "mode"],
                    help="Choose how to handle missing values"
                )
                
                columns_to_process = st.multiselect(
                    "Columns to process (leave empty for all):",
                    missing_cols,
                    default=missing_cols
                )
                
                if st.button("Apply Missing Value Strategy"):
                    try:
                        cols = columns_to_process if columns_to_process else None
                        st.session_state.processed_data = processor.handle_missing_values(
                            st.session_state.processed_data,
                            strategy=strategy,
                            columns=cols
                        )
                        st.success(f"✅ Applied '{strategy}' strategy successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.info("No missing values to handle!")
        
        with col2:
            st.markdown("### 🏷️ Encode Categorical Variables")
            
            # Get categorical columns
            cat_cols = st.session_state.processed_data.select_dtypes(
                include=['object']
            ).columns.tolist()
            
            if len(cat_cols) > 0:
                encoding_method = st.selectbox(
                    "Encoding method:",
                    ["label", "onehot"],
                    help="Label: Convert to numbers | One-hot: Create binary columns"
                )
                
                cols_to_encode = st.multiselect(
                    "Columns to encode:",
                    cat_cols,
                    default=cat_cols
                )
                
                if st.button("Apply Encoding"):
                    try:
                        st.session_state.processed_data = processor.encode_categorical(
                            st.session_state.processed_data,
                            method=encoding_method,
                            columns=cols_to_encode
                        )
                        st.success(f"✅ Applied '{encoding_method}' encoding successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.info("No categorical columns to encode!")
        
        st.write("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### 📏 Scale Features")
            
            # Get numerical columns
            num_cols = st.session_state.processed_data.select_dtypes(
                include=['float64', 'int64']
            ).columns.tolist()
            
            if len(num_cols) > 0:
                scaling_method = st.selectbox(
                    "Scaling method:",
                    ["standard", "minmax"],
                    help="Standard: Mean=0, Std=1 | MinMax: Scale to [0,1]"
                )
                
                cols_to_scale = st.multiselect(
                    "Columns to scale:",
                    num_cols,
                    help="Select columns to scale (typically exclude target variable)"
                )
                
                if st.button("Apply Scaling"):
                    try:
                        st.session_state.processed_data = processor.scale_features(
                            st.session_state.processed_data,
                            method=scaling_method,
                            columns=cols_to_scale
                        )
                        st.success(f"✅ Applied '{scaling_method}' scaling successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.info("No numerical columns to scale!")
        
        with col4:
            st.markdown("### 🔍 Remove Outliers")
            
            num_cols = st.session_state.processed_data.select_dtypes(
                include=['float64', 'int64']
            ).columns.tolist()
            
            if len(num_cols) > 0:
                outlier_method = st.selectbox(
                    "Detection method:",
                    ["iqr", "zscore"],
                    help="IQR: Interquartile Range | Z-score: Standard deviations from mean"
                )
                
                cols_for_outliers = st.multiselect(
                    "Columns to check:",
                    num_cols
                )
                
                if cols_for_outliers and st.button("Remove Outliers"):
                    try:
                        original_shape = st.session_state.processed_data.shape[0]
                        st.session_state.processed_data = processor.remove_outliers(
                            st.session_state.processed_data,
                            columns=cols_for_outliers,
                            method=outlier_method
                        )
                        new_shape = st.session_state.processed_data.shape[0]
                        removed = original_shape - new_shape
                        st.success(f"✅ Removed {removed} outlier rows!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.info("No numerical columns available!")
        
        st.write("---")
        
        # Show current data stats after preprocessing
        st.markdown("### 📊 Current Data Statistics")
        display_data_stats(st.session_state.processed_data)
    
    # Tab 3: Visualizations
    with tab3:
        st.subheader("Data Visualizations")
        
        viz_type = st.selectbox(
            "Select visualization type:",
            [
                "Correlation Heatmap",
                "Distribution Plot (Numerical)",
                "Count Plot (Categorical)",
                "Data Overview Dashboard"
            ]
        )
        
        if viz_type == "Correlation Heatmap":
            st.markdown("#### 🔥 Correlation Heatmap")
            st.info("Shows correlation between numerical features. Values close to 1 or -1 indicate strong relationships.")
            visualizer.plot_correlation_heatmap(st.session_state.processed_data)
        
        elif viz_type == "Distribution Plot (Numerical)":
            num_cols = st.session_state.processed_data.select_dtypes(
                include=['float64', 'int64']
            ).columns.tolist()
            
            if len(num_cols) > 0:
                selected_col = st.selectbox("Select column:", num_cols)
                bins = st.slider("Number of bins:", 10, 100, 30)
                
                st.markdown(f"#### 📊 Distribution of {selected_col}")
                visualizer.plot_distribution(
                    st.session_state.processed_data,
                    selected_col,
                    bins=bins
                )
            else:
                st.warning("No numerical columns available!")
        
        elif viz_type == "Count Plot (Categorical)":
            cat_cols = st.session_state.processed_data.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()
            
            if len(cat_cols) > 0:
                selected_col = st.selectbox("Select column:", cat_cols)
                top_n = st.slider("Show top N categories:", 5, 50, 20)
                
                st.markdown(f"#### 📊 Distribution of {selected_col}")
                visualizer.plot_categorical_distribution(
                    st.session_state.processed_data,
                    selected_col,
                    top_n=top_n
                )
            else:
                st.warning("No categorical columns available!")
        
        elif viz_type == "Data Overview Dashboard":
            st.markdown("#### 📊 Data Overview Dashboard")
            visualizer.plot_data_overview(st.session_state.processed_data)
    
    # Tab 4: Train/Test Split
    with tab4:
        st.subheader("Split Data for Training")
        
        st.markdown("""
        Before training models, split your data into training and testing sets.
        The training set is used to train the model, while the test set evaluates its performance.
        """)
        
        # Select target column
        st.markdown("#### 🎯 Select Target Variable")
        target_column = st.selectbox(
            "Target column (what you want to predict):",
            st.session_state.processed_data.columns.tolist(),
            key="target_select"
        )
        
        # Detect task type
        if target_column:
            task_type = detect_task_type(st.session_state.processed_data[target_column])
            st.info(f"📊 Detected task type: **{task_type.upper()}**")
            st.session_state.task_type = task_type
        
        # Select features
        available_features = [col for col in st.session_state.processed_data.columns 
                             if col != target_column]
        
        st.markdown("#### 🔢 Select Features")
        feature_columns = st.multiselect(
            "Feature columns (used for prediction):",
            available_features,
            default=available_features,
            key="feature_select"
        )
        
        # Test size
        st.markdown("#### ⚖️ Split Ratio")
        test_size = st.slider(
            "Test set size (%):",
            10, 50, 20,
            help="Percentage of data to use for testing"
        ) / 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Set", f"{(1-test_size)*100:.0f}%")
        with col2:
            st.metric("Test Set", f"{test_size*100:.0f}%")
        
        # Split button
        if st.button("🔪 Split Data", type="primary"):
            if not feature_columns:
                st.error("❌ Please select at least one feature column!")
            else:
                try:
                    # Create dataset with selected features and target
                    data_for_split = st.session_state.processed_data[feature_columns + [target_column]]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = processor.split_data(
                        data_for_split,
                        target_column=target_column,
                        test_size=test_size,
                        random_state=42
                    )
                    
                    # Store in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.target_column = target_column
                    st.session_state.feature_columns = feature_columns
                    
                    st.success("✅ Data split successfully!")
                    
                    # Show split statistics
                    st.write("---")
                    st.markdown("#### 📊 Split Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Train Samples", len(X_train))
                    with col2:
                        st.metric("Test Samples", len(X_test))
                    with col3:
                        st.metric("Features", len(feature_columns))
                    with col4:
                        st.metric("Target", target_column)
                    
                    st.info("✨ You're ready to train models! Go to the **ML Training** page.")
                    
                except Exception as e:
                    st.error(f"❌ Error splitting data: {str(e)}")
        
        # Show current split status
        if st.session_state.X_train is not None:
            st.write("---")
            st.markdown("#### ✅ Current Split Status")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Train:** {st.session_state.X_train.shape[0]} samples")
            with col2:
                st.info(f"**Test:** {st.session_state.X_test.shape[0]} samples")
            with col3:
                st.info(f"**Features:** {len(st.session_state.feature_columns)}")
