"""
Feature engineering module for enhanced churn prediction.
Creates new features based on domain knowledge and data exploration.
"""

import pandas as pd
import numpy as np

def create_tenure_groups(df):
    """Create tenure groups for better segmentation."""
    df_features = df.copy()
    
    # Create tenure groups
    def tenure_group(tenure):
        if tenure <= 12:
            return 'New'
        elif tenure <= 24:
            return 'Medium'
        elif tenure <= 48:
            return 'Long'
        else:
            return 'Very_Long'
    
    if 'tenure' in df_features.columns:
        df_features['TenureGroup'] = df_features['tenure'].apply(tenure_group)
    
    return df_features

def create_spending_features(df):
    """Create spending-related features."""
    df_features = df.copy()
    
    if 'MonthlyCharges' in df_features.columns and 'tenure' in df_features.columns:
        # Average monthly charges per tenure month
        df_features['AvgMonthlyCharges'] = df_features['MonthlyCharges'] / (df_features['tenure'] + 1)
        
        # Total charges to monthly charges ratio
        if 'TotalCharges' in df_features.columns:
            df_features['TotalToMonthlyRatio'] = df_features['TotalCharges'] / df_features['MonthlyCharges']
            
            # Customer Lifetime Value approximation
            df_features['CLV'] = df_features['TotalCharges'] - (df_features['MonthlyCharges'] * df_features['tenure'])
    
    return df_features

def create_service_features(df):
    """Create service-related features."""
    df_features = df.copy()
    
    # Count of additional services
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    existing_service_cols = [col for col in service_cols if col in df_features.columns]
    if existing_service_cols:
        # Convert 'Yes'/'No' to 1/0 for counting
        for col in existing_service_cols:
            if df_features[col].dtype == 'object':
                df_features[col + '_Binary'] = (df_features[col] == 'Yes').astype(int)
        
        # Count total services
        binary_cols = [col + '_Binary' for col in existing_service_cols]
        existing_binary_cols = [col for col in binary_cols if col in df_features.columns]
        
        if existing_binary_cols:
            df_features['TotalServices'] = df_features[existing_binary_cols].sum(axis=1)
    
    return df_features

def create_contract_features(df):
    """Create contract-related features."""
    df_features = df.copy()
    
    if 'Contract' in df_features.columns:
        # Contract commitment level
        contract_mapping = {
            'Month-to-month': 0,
            'One year': 1,
            'Two year': 2
        }
        df_features['ContractCommitment'] = df_features['Contract'].map(contract_mapping)
        
        # Is flexible contract (month-to-month)
        df_features['IsFlexibleContract'] = (df_features['Contract'] == 'Month-to-month').astype(int)
    
    return df_features

def create_customer_segments(df):
    """Create customer segments based on multiple features."""
    df_features = df.copy()
    
    # High value customer (high monthly charges and long tenure)
    if 'MonthlyCharges' in df_features.columns and 'tenure' in df_features.columns:
        high_charges = df_features['MonthlyCharges'] > df_features['MonthlyCharges'].quantile(0.75)
        long_tenure = df_features['tenure'] > df_features['tenure'].quantile(0.75)
        df_features['HighValueCustomer'] = (high_charges & long_tenure).astype(int)
        
        # At-risk segment (high charges but short tenure)
        short_tenure = df_features['tenure'] <= df_features['tenure'].quantile(0.25)
        df_features['AtRiskCustomer'] = (high_charges & short_tenure).astype(int)
    
    return df_features

def create_payment_features(df):
    """Create payment-related features."""
    df_features = df.copy()
    
    if 'PaymentMethod' in df_features.columns:
        # Is electronic payment
        electronic_methods = ['Electronic check', 'Credit card (automatic)', 'Bank transfer (automatic)']
        df_features['IsElectronicPayment'] = df_features['PaymentMethod'].isin(electronic_methods).astype(int)
        
        # Is automatic payment
        automatic_methods = ['Credit card (automatic)', 'Bank transfer (automatic)']
        df_features['IsAutomaticPayment'] = df_features['PaymentMethod'].isin(automatic_methods).astype(int)
    
    return df_features

def create_all_features(df):
    """Create all engineered features."""
    print("Creating engineered features...")
    
    # Apply all feature engineering functions
    df_features = create_tenure_groups(df)
    df_features = create_spending_features(df_features)
    df_features = create_service_features(df_features)
    df_features = create_contract_features(df_features)
    df_features = create_customer_segments(df_features)
    df_features = create_payment_features(df_features)
    
    print(f"Feature engineering completed. New shape: {df_features.shape}")
    return df_features

def get_feature_importance_names():
    """Return list of engineered feature names for interpretation."""
    return [
        'TenureGroup',
        'AvgMonthlyCharges',
        'TotalToMonthlyRatio',
        'CLV',
        'TotalServices',
        'ContractCommitment',
        'IsFlexibleContract',
        'HighValueCustomer',
        'AtRiskCustomer',
        'IsElectronicPayment',
        'IsAutomaticPayment'
    ]
