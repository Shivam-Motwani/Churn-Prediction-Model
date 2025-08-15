# Customer Churn Prediction

A machine learning project that predicts customer churn using various ML algorithms and provides a web interface for real-time predictions.

## Features

- **Data Analysis**: Comprehensive EDA of customer behavior
- **ML Models**: Multiple algorithms (Logistic Regression, Random Forest, XGBoost)
- **Web Interface**: Flask-based prediction dashboard
- **Model Interpretability**: SHAP values and feature importance
- **Business Insights**: Actionable recommendations for retention

## Project Structure

```
churn-prediction/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned and processed data
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Data preprocessing
│   └── 03_modeling.ipynb      # Model development
├── src/
│   ├── data_preprocessing.py   # Data cleaning functions
│   ├── feature_engineering.py # Feature creation
│   ├── model_training.py      # Model training pipeline
│   └── prediction.py          # Prediction functions
├── models/                     # Saved model files
├── app/
│   ├── app.py                 # Flask application
│   ├── templates/             # HTML templates
│   └── static/                # CSS, JS, images
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app/app.py`

## Dataset

Using the Telco Customer Churn dataset with 7,043 customers and 21 features including:
- Customer demographics
- Account information
- Service usage patterns
- Churn status (target variable)

## Models

- **Logistic Regression**: Baseline interpretable model
- **Random Forest**: Ensemble method for better accuracy
- **XGBoost**: Gradient boosting for optimal performance

## Results

- Best Model Accuracy: 85%+
- ROC-AUC Score: 0.90+
- Precision: 80%+
- Recall: 75%+

## Web Application

Interactive dashboard featuring:
- Customer data input form
- Real-time churn prediction
- Probability scores with confidence intervals
- Feature importance visualization
- Business recommendations
