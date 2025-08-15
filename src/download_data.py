"""
Dataset Download Script for Telco Customer Churn

This script downloads the Telco Customer Churn dataset for analysis.
"""

import os
import pandas as pd
import requests
from urllib.parse import urlparse

def download_telco_dataset():
    """
    Download the Telco Customer Churn dataset
    """
    # Create data directories
    os.makedirs('../data/raw', exist_ok=True)
    os.makedirs('../data/processed', exist_ok=True)
    
    # Dataset URL (this is a sample - you'll need to get the actual dataset from Kaggle)
    # For Kaggle datasets, you typically need to use the Kaggle API
    print("To download the Telco Customer Churn dataset:")
    print("1. Go to: https://www.kaggle.com/blastchar/telco-customer-churn")
    print("2. Download the dataset manually")
    print("3. Place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in the ../data/raw/ folder")
    
    # Create a sample dataset for testing
    create_sample_dataset()

def create_sample_dataset():
    """
    Create a sample dataset for testing purposes
    """
    import numpy as np
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample data
    data = {
        'customerID': [f'CUST{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.52, 0.48]),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70]),
        'tenure': np.random.randint(1, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.91, 0.09]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.42, 0.49, 0.09]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.28, 0.50, 0.22]),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.44, 0.22]),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.44, 0.22]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.29, 0.49, 0.22]),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.38, 0.40, 0.22]),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.39, 0.39, 0.22]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.21, 0.24]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41]),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples, p=[0.34, 0.19, 0.22, 0.25]),
        'MonthlyCharges': np.random.uniform(18.25, 118.75, n_samples),
        'TotalCharges': np.random.uniform(18.80, 8684.80, n_samples),
    }
    
    # Create churn target with some logic
    churn_prob = np.random.random(n_samples)
    
    # Adjust probabilities based on features
    for i in range(n_samples):
        if data['Contract'][i] == 'Month-to-month':
            churn_prob[i] += 0.3
        if data['tenure'][i] < 12:
            churn_prob[i] += 0.2
        if data['MonthlyCharges'][i] > 80:
            churn_prob[i] += 0.1
        if data['SeniorCitizen'][i] == 1:
            churn_prob[i] += 0.1
    
    data['Churn'] = ['Yes' if prob > 0.5 else 'No' for prob in churn_prob]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save sample dataset
    output_path = '../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Sample dataset created: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
    
    return df

if __name__ == "__main__":
    download_telco_dataset()
