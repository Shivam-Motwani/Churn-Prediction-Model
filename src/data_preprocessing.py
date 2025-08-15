"""
Data preprocessing module for churn prediction.
Handles data cleaning, encoding, and feature preparation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load the churn dataset from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

def clean_data(df):
    """Clean the dataset by handling missing values and data types."""
    # Create a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Convert TotalCharges to numeric (handle spaces as NaN)
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Fill missing values in TotalCharges with median
    df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)
    
    # Convert binary categorical variables to 0/1
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map({'No': 0, 'Yes': 1, 'Female': 0, 'Male': 1})
    
    print(f"Data cleaned. Shape: {df_clean.shape}")
    return df_clean

def encode_categorical(df):
    """Encode categorical variables using label encoding."""
    df_encoded = df.copy()
    
    # Categorical columns to encode
    categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaymentMethod']
    
    # Initialize label encoder
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
    
    return df_encoded, label_encoders

def prepare_features(df):
    """Prepare features for modeling."""
    # Remove customerID as it's not useful for prediction
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Separate features and target
    if 'Churn' in df.columns:
        X = df.drop('Churn', axis=1)
        y = df['Churn']
    else:
        X = df
        y = None
    
    return X, y

def scale_features(X_train, X_test=None):
    """Scale numerical features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def preprocess_pipeline(filepath):
    """Complete preprocessing pipeline."""
    # Load data
    df = load_data(filepath)
    if df is None:
        return None
    
    # Clean data
    df_clean = clean_data(df)
    
    # Encode categorical variables
    df_encoded, encoders = encode_categorical(df_clean)
    
    # Prepare features
    X, y = prepare_features(df_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'scaler': scaler,
        'encoders': encoders
    }
