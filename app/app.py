"""
Flask web application for churn prediction.
Provides interactive interface for customer churn prediction.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
app.secret_key = 'churn-prediction-secret-key-change-in-production'

# Global variables for loaded model and preprocessors
model = None
scaler = None
label_encoders = None
target_encoder = None
feature_names = None

def load_model_and_preprocessors():
    """Load the trained model and preprocessors."""
    global model, scaler, label_encoders, target_encoder, feature_names
    
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    try:
        # Try to load existing model files
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.startswith('best_churn_model') and f.endswith('.pkl')]
            
            if model_files:
                model_path = os.path.join(models_dir, model_files[0])
                model = joblib.load(model_path)
                
                # Load preprocessing objects
                scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
                label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
                target_encoder = joblib.load(os.path.join(models_dir, 'target_encoder.pkl'))
                feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
                
                print(f"Model loaded successfully: {model_files[0]}")
                return
        
        raise FileNotFoundError("No trained model found")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training a simple model for demonstration...")
        train_simple_model()

def train_simple_model():
    """Train a simple model for demonstration"""
    global model, scaler, label_encoders, target_encoder, feature_names
    
    print("Creating training data...")
    
    # Create sample data for training
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic sample data
    data = {
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        'partner': np.random.choice(['Yes', 'No'], n_samples),
        'dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(1, 73, n_samples),
        'phone_service': np.random.choice(['Yes', 'No'], n_samples),
        'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
        'payment_method': np.random.choice([
            'Electronic check', 'Mailed check', 
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples),
        'monthly_charges': np.random.uniform(18.25, 118.75, n_samples),
        'total_charges': np.random.uniform(18.80, 8684.80, n_samples),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create target variable with some logic
    churn_prob = np.random.random(n_samples) * 0.3  # Base probability
    
    # Adjust probabilities based on features
    for i in range(n_samples):
        if df.loc[i, 'contract'] == 'Month-to-month':
            churn_prob[i] += 0.4
        if df.loc[i, 'tenure'] < 12:
            churn_prob[i] += 0.3
        if df.loc[i, 'monthly_charges'] > 80:
            churn_prob[i] += 0.2
        if df.loc[i, 'senior_citizen'] == 1:
            churn_prob[i] += 0.1
        if df.loc[i, 'payment_method'] == 'Electronic check':
            churn_prob[i] += 0.2
    
    df['churn'] = (churn_prob > 0.5).astype(int)
    
    print("Preprocessing data...")
    
    # Preprocessing
    # Encode categorical variables
    label_encoders = {}
    for col in ['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines',
                'internet_service', 'online_security', 'online_backup', 'device_protection',
                'tech_support', 'streaming_tv', 'streaming_movies', 'contract',
                'paperless_billing', 'payment_method']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Prepare features and target
    X = df.drop('churn', axis=1)
    y = df['churn']
    feature_names = list(X.columns)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['tenure', 'monthly_charges', 'total_charges']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    print("Training model...")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create target encoder
    target_encoder = LabelEncoder()
    target_encoder.fit(['No', 'Yes'])
    
    # Save model and preprocessing objects
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(models_dir, 'best_churn_model_random_forest.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(label_encoders, os.path.join(models_dir, 'label_encoders.pkl'))
    joblib.dump(target_encoder, os.path.join(models_dir, 'target_encoder.pkl'))
    joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.pkl'))
    
    print("Simple model trained and saved!")

def validate_total_charges(tenure, monthly_charges, total_charges):
    """
    Validate and potentially correct total charges based on tenure and monthly charges
    """
    try:
        tenure = float(tenure) if tenure else 0
        monthly_charges = float(monthly_charges) if monthly_charges else 0
        total_charges = float(total_charges) if total_charges else 0
        
        # Calculate expected total charges (tenure * monthly charges)
        expected_total = tenure * monthly_charges
        
        # If total charges is 0 or significantly different from expected, use calculated value
        if total_charges == 0 or abs(total_charges - expected_total) > (expected_total * 0.5):
            print(f"Total charges validation: Expected {expected_total:.2f}, got {total_charges:.2f}. Using calculated value.")
            return expected_total
        
        return total_charges
    except (ValueError, TypeError):
        return 0.0

def preprocess_input(form_data):
    """Preprocess form input for model prediction"""
    try:
        # Create a dictionary with all expected features
        processed_data = {}
        
        # Map form fields to model features
        field_mapping = {
            'gender': 'gender',
            'senior_citizen': 'senior_citizen',
            'partner': 'partner',
            'dependents': 'dependents',
            'tenure': 'tenure',
            'phone_service': 'phone_service',
            'multiple_lines': 'multiple_lines',
            'internet_service': 'internet_service',
            'online_security': 'online_security',
            'online_backup': 'online_backup',
            'device_protection': 'device_protection',
            'tech_support': 'tech_support',
            'streaming_tv': 'streaming_tv',
            'streaming_movies': 'streaming_movies',
            'contract': 'contract',
            'paperless_billing': 'paperless_billing',
            'payment_method': 'payment_method',
            'monthly_charges': 'monthly_charges',
            'total_charges': 'total_charges'
        }
        
        # Extract and convert data
        for form_field, model_field in field_mapping.items():
            if form_field in form_data:
                value = form_data[form_field]
                
                # Handle numerical fields with proper conversion
                if form_field in ['tenure', 'monthly_charges', 'total_charges']:
                    try:
                        processed_data[model_field] = float(value) if value else 0.0
                    except (ValueError, TypeError):
                        processed_data[model_field] = 0.0
                elif form_field == 'senior_citizen':
                    try:
                        processed_data[model_field] = int(value) if value else 0
                    except (ValueError, TypeError):
                        processed_data[model_field] = 0
                else:
                    processed_data[model_field] = str(value) if value else 'No'
            else:
                # Set default values for missing fields
                if form_field in ['tenure', 'monthly_charges', 'total_charges']:
                    processed_data[model_field] = 0.0
                elif form_field == 'senior_citizen':
                    processed_data[model_field] = 0
                else:
                    processed_data[model_field] = 'No'
        
        # Validate and correct total charges
        if 'tenure' in processed_data and 'monthly_charges' in processed_data and 'total_charges' in processed_data:
            processed_data['total_charges'] = validate_total_charges(
                processed_data['tenure'], 
                processed_data['monthly_charges'], 
                processed_data['total_charges']
            )
        
        print(f"Form data received: {dict(form_data)}")
        print(f"Processed data: {processed_data}")
        
        # Create DataFrame
        df_input = pd.DataFrame([processed_data])
        
        # Apply label encoding for categorical variables
        for col in df_input.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                try:
                    df_input[col] = label_encoders[col].transform(df_input[col])
                except ValueError:
                    # Handle unseen categories by using the most common class (0)
                    df_input[col] = 0
        
        # Scale numerical features
        numerical_cols = ['tenure', 'monthly_charges', 'total_charges']
        existing_numerical_cols = [col for col in numerical_cols if col in df_input.columns]
        if existing_numerical_cols and scaler is not None:
            df_input[existing_numerical_cols] = scaler.transform(df_input[existing_numerical_cols])
        
        # Ensure all required features are present and in correct order
        if feature_names:
            # Reorder columns to match training data
            df_input = df_input.reindex(columns=feature_names, fill_value=0)
        
        # Convert to float to ensure all values are numeric
        df_input = df_input.astype(float)
        
        return df_input
    
    except Exception as e:
        print(f"Error preprocessing input: {e}")
        return None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'POST':
        try:
            print(f"Received form data: {dict(request.form)}")
            
            # Preprocess input
            processed_input = preprocess_input(request.form)
            
            if processed_input is None:
                raise ValueError("Invalid input data")
            
            print(f"Processed input shape: {processed_input.shape}")
            print(f"Processed input type: {type(processed_input)}")
            print(f"Processed input columns: {processed_input.columns.tolist()}")
            
            # Make prediction using DataFrame to preserve feature names
            prediction_proba = model.predict_proba(processed_input)[0][1]
            prediction = int(prediction_proba > 0.5)
            
            print(f"Prediction probability: {prediction_proba}")
            print(f"Prediction: {prediction}")
            
            # Determine risk level
            if prediction_proba < 0.3:
                risk_level = 'low'
            elif prediction_proba < 0.7:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            # Get feature importance (top 5)
            if hasattr(model, 'feature_importances_') and feature_names:
                importance_pairs = list(zip(feature_names, model.feature_importances_))
                importance_pairs.sort(key=lambda x: x[1], reverse=True)
                feature_importance = importance_pairs[:5]
            else:
                feature_importance = []
            
            # Prepare customer data for display with proper type conversion
            customer_data = dict(request.form)
            
            # Convert numeric fields to proper types for display
            numeric_fields = ['tenure', 'monthly_charges', 'total_charges']
            for field in numeric_fields:
                if field in customer_data:
                    try:
                        customer_data[field] = float(customer_data[field])
                    except (ValueError, TypeError):
                        customer_data[field] = 0.0
            
            return render_template('result.html',
                                prediction=prediction,
                                prediction_proba=prediction_proba,
                                risk_level=risk_level,
                                feature_importance=feature_importance,
                                customer_data=customer_data)
                                
        except Exception as e:
            print(f"Prediction error: {e}")
            return render_template('predict.html', error="An error occurred during prediction. Please try again.")
    
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        # Preprocess input
        processed_input = preprocess_input(data)
        
        if processed_input is None:
            return jsonify({'error': 'Invalid input data'}), 400
        
        # Make prediction - handle both DataFrame and array inputs
        if hasattr(processed_input, 'values'):
            prediction_proba = model.predict_proba(processed_input.values)[0][1]
        else:
            prediction_proba = model.predict_proba(processed_input)[0][1]
        
        prediction = int(prediction_proba > 0.5)
        
        # Determine risk level
        if prediction_proba < 0.3:
            risk_level = 'low'
        elif prediction_proba < 0.7:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return jsonify({
            'churn_probability': float(prediction_proba),
            'churn_prediction': prediction,
            'risk_level': risk_level
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Loading model and starting Flask app...")
    load_model_and_preprocessors()
    print("Model loaded successfully!")
    print("Starting Flask app on http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
