# 🧪 AI Lab Studio

A comprehensive, production-ready Streamlit application for end-to-end machine learning development. Built for data scientists, ML engineers, and anyone looking to rapidly prototype and deploy machine learning models.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### 📊 Data Hub
- **Data Loading**: Upload CSV files or use built-in sample datasets
- **Data Exploration**: Interactive data preview with pagination, statistics, and metadata
- **Data Preprocessing**:
  - Handle missing values (drop, mean, median, mode)
  - Encode categorical variables (label encoding, one-hot encoding)
  - Feature scaling (StandardScaler, MinMaxScaler)
  - Outlier detection and removal
- **Visualizations**:
  - Correlation heatmaps
  - Distribution plots for numerical features
  - Count plots for categorical features
  - Interactive data overview dashboard
- **Train/Test Split**: Customizable split ratios with automatic data preparation

### 🤖 ML Training
- **Multiple Algorithms**:
  - Classification: Logistic Regression, Random Forest, XGBoost, SVM
  - Regression: Linear Regression, Random Forest, XGBoost, SVM
- **Hyperparameter Tuning**: Interactive controls for each algorithm
- **Feature Selection**: Choose which features to use for training
- **Model Evaluation**:
  - Classification: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report
  - Regression: MAE, MSE, RMSE, R², Residual Plots
- **Feature Importance**: Visualizations for tree-based models
- **Model Comparison**: Compare performance across multiple trained models

### 🎯 Prediction
- **Single Prediction**: Interactive form with intelligent input validation
- **Batch Prediction**: Upload CSV files for bulk predictions
- **Confidence Scores**: Probability distributions for classification tasks
- **Results Export**: Download predictions as CSV files
- **Sample Templates**: Generate prediction templates with correct format

### 💾 Project Management
- **Model Persistence**: Save trained models to disk with metadata
- **Model Loading**: Load previously saved models for inference
- **Model Registry**: View all saved models with performance metrics
- **Export Capabilities**: Export model registry as JSON or CSV
- **Model Management**: Delete individual or bulk models
- **Statistics Dashboard**: Overview of your model collection

## 🚀 Quick Start

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run app.py
```

4. **Open your browser** to `http://localhost:8501`

### First Steps

1. **Load Data**: Go to the Data Hub page and load the Iris sample dataset
2. **Explore**: View statistics, visualizations, and data quality metrics
3. **Preprocess**: Handle any data issues (usually none with sample datasets)
4. **Split Data**: Create train/test split (80/20 is recommended)
5. **Train Model**: Navigate to ML Training, select an algorithm, and train
6. **Evaluate**: Review performance metrics and feature importance
7. **Predict**: Use the Prediction page for single or batch predictions
8. **Save**: Go to Project Management to save your trained model

## 📁 Project Structure

```
ai_lab_studio/
├── app.py                          # Main application entry point
├── pages/
│   ├── 1_Data_Hub.py              # Data loading and preprocessing
│   ├── 2_ML_Training.py           # Model training and evaluation
│   ├── 3_Prediction.py            # Single and batch predictions
│   └── 4_Project_Management.py    # Model saving and management
├── utils/
│   ├── data_processor.py          # Data processing utilities
│   ├── ml_models.py               # ML model training and evaluation
│   ├── visualizations.py          # Visualization functions
│   └── helpers.py                 # Helper functions and utilities
├── sample_data/
│   ├── iris.csv                   # Sample classification dataset
│   └── boston_housing.csv         # Sample regression dataset
├── models/                         # Saved models directory
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 📊 Sample Datasets

Two high-quality datasets are included for immediate experimentation:

### 1. Iris Dataset (Classification)
- **Samples**: 150
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Target**: Species (setosa, versicolor, virginica)
- **Use Case**: Multi-class classification

### 2. California Housing Dataset (Regression)
- **Samples**: 20,640
- **Features**: 8 (median income, house age, rooms, etc.)
- **Target**: Median house value
- **Use Case**: Regression prediction

## 🔧 Technical Details

### Algorithms Supported

#### Classification
- **Logistic Regression**: Fast, interpretable linear classifier
- **Random Forest**: Robust ensemble method with feature importance
- **XGBoost**: State-of-the-art gradient boosting
- **SVM**: Support Vector Machine with multiple kernels

#### Regression
- **Linear Regression**: Simple, interpretable linear model
- **Random Forest**: Ensemble method for non-linear relationships
- **XGBoost**: Powerful gradient boosting for regression
- **SVM**: Support Vector Regression with kernel methods

### Performance Optimizations

- **Session State Management**: Efficient data persistence across pages
- **Caching**: `@st.cache_data` and `@st.cache_resource` for performance
- **Lazy Loading**: Data and models loaded only when needed
- **Pagination**: Large datasets displayed with pagination

### Error Handling

- Comprehensive try-catch blocks throughout the application
- User-friendly error messages
- Input validation and data quality checks
- Graceful degradation for missing features

## 📚 Usage Guide

### Data Preprocessing Tips

1. **Missing Values**: Start with 'drop' for small amounts, use 'mean/median' for numerical features
2. **Encoding**: Use 'label' encoding for ordinal categories, 'onehot' for nominal categories
3. **Scaling**: Always scale features before training (except for tree-based models)
4. **Outliers**: Remove carefully - they might be important for your problem

### Model Selection Guide

- **Small datasets (<1000 samples)**: Start with Logistic/Linear Regression
- **Medium datasets**: Random Forest is usually a solid choice
- **Large datasets**: XGBoost for best performance
- **Interpretability**: Use Logistic/Linear Regression or Random Forest
- **Best performance**: Try XGBoost and ensemble methods

### Hyperparameter Tuning

- **Random Forest**: Increase `n_estimators` for better performance, adjust `max_depth` to prevent overfitting
- **XGBoost**: Lower `learning_rate` with higher `n_estimators` for better results
- **SVM**: Try different kernels, adjust `C` for regularization strength

## 🛠️ Development

### Requirements

- Python 3.8 or higher
- See `requirements.txt` for package dependencies

### Adding New Features

The modular structure makes it easy to extend:

1. **New algorithms**: Add to `utils/ml_models.py`
2. **New visualizations**: Add to `utils/visualizations.py`
3. **New preprocessing**: Add to `utils/data_processor.py`
4. **New pages**: Create in `pages/` directory with naming convention `N_Page_Name.py`

### Testing

Test the application with the sample datasets:

```bash
# Run the app
streamlit run app.py

# Test workflow
1. Load Iris dataset
2. Split data (80/20)
3. Train Random Forest classifier
4. Check accuracy > 0.90
5. Make predictions
6. Save model
7. Load model
8. Verify predictions match
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional ML algorithms (Neural Networks, Naive Bayes, etc.)
- Advanced feature engineering
- Automated hyperparameter tuning (Grid Search, Random Search)
- Model interpretability (SHAP, LIME)
- Data augmentation techniques
- Export to production formats (ONNX, TensorFlow Lite)

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- ML powered by [scikit-learn](https://scikit-learn.org/) and [XGBoost](https://xgboost.readthedocs.io/)
- Visualizations using [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), and [Plotly](https://plotly.com/)

## 📞 Support

For issues, questions, or contributions:

- Open an issue on the repository
- Check the documentation in the app's expandable sections
- Review the code comments for implementation details

## 🎯 Roadmap

- [ ] Add Neural Network support (TensorFlow/PyTorch)
- [ ] Implement AutoML capabilities
- [ ] Add time series forecasting
- [ ] Support for image classification
- [ ] Model deployment to cloud platforms
- [ ] A/B testing framework
- [ ] Real-time predictions API

---

**Built with ❤️ for the ML Community**

Made with [Streamlit](https://streamlit.io/) | Powered by Python 🐍
