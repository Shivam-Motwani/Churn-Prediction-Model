#!/usr/bin/env python3
"""
Quick test of the churn model functionality
"""
import requests
import json
import time

def test_churn_model():
    print("🧪 Testing Churn Model...")
    
    # Wait for Flask to be ready
    time.sleep(2)
    
    # Test data
    customer_data = {
        'gender': 'Female',
        'senior_citizen': '0',
        'partner': 'Yes',
        'dependents': 'No',
        'tenure': '12',
        'phone_service': 'Yes',
        'multiple_lines': 'No',
        'internet_service': 'DSL',
        'online_security': 'No',
        'online_backup': 'Yes',
        'device_protection': 'No',
        'tech_support': 'No',
        'streaming_tv': 'No',
        'streaming_movies': 'No',
        'contract': 'Month-to-month',
        'paperless_billing': 'Yes',
        'payment_method': 'Electronic check',
        'monthly_charges': '65.50',
        'total_charges': '785.50'
    }
    
    try:
        # Test API endpoint
        print("📡 Testing API endpoint...")
        response = requests.post('http://127.0.0.1:5000/api/predict', 
                               json=customer_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Test SUCCESS!")
            print(f"   Churn Probability: {result['churn_probability']:.1%}")
            print(f"   Risk Level: {result['risk_level'].upper()}")
            print(f"   Prediction: {'Will Churn' if result['churn_prediction'] == 1 else 'Will Stay'}")
        else:
            print(f"❌ API Test FAILED: {response.status_code}")
            
    except Exception as e:
        print(f"❌ API Test ERROR: {e}")
    
    try:
        # Test web form
        print("\n📝 Testing Web Form...")
        response = requests.post('http://127.0.0.1:5000/predict', 
                               data=customer_data, timeout=15)
        
        print(f"   Status: {response.status_code}")
        print(f"   Content Length: {len(response.text)} chars")
        
        if 'Churn Prediction Result' in response.text:
            print("✅ Web Form Test SUCCESS!")
            
            # Extract probability
            import re
            prob_matches = re.findall(r'(\d+\.\d+)%', response.text)
            if prob_matches:
                print(f"   Churn Probability: {prob_matches[0]}%")
                
        elif 'An error occurred during prediction' in response.text:
            print("❌ Web Form Test FAILED: Error in prediction")
        elif 'Predict Customer Churn' in response.text:
            print("⚠️ Web Form Test: Still on form page (validation issue)")
        else:
            print("❓ Web Form Test: Unexpected response")
            
    except Exception as e:
        print(f"❌ Web Form Test ERROR: {e}")

if __name__ == "__main__":
    test_churn_model()
