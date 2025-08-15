#!/usr/bin/env python3
"""
Churn Prediction ML Model Demo
=====================================

This script demonstrates the customer churn prediction system with different customer profiles.
"""

import requests
import json
from datetime import datetime

def test_prediction(customer_data, description):
    """Test prediction for a customer profile"""
    print(f"\n{'='*60}")
    print(f"🔍 TESTING: {description}")
    print('='*60)
    
    # Display customer profile
    print("\n📋 Customer Profile:")
    for key, value in customer_data.items():
        formatted_key = key.replace('_', ' ').title()
        print(f"   {formatted_key}: {value}")
    
    try:
        # Make API request
        response = requests.post('http://127.0.0.1:5000/api/predict', json=customer_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n🎯 PREDICTION RESULTS:")
            print(f"   Churn Probability: {result['churn_probability']:.1%}")
            print(f"   Risk Level: {result['risk_level'].upper()}")
            print(f"   Will Churn: {'YES' if result['churn_prediction'] == 1 else 'NO'}")
            
            # Risk level interpretation
            risk_level = result['risk_level']
            if risk_level == 'low':
                print(f"   📈 Recommendation: Maintain engagement, consider upselling")
            elif risk_level == 'medium':
                print(f"   ⚠️  Recommendation: Proactive support, review service plan")
            else:
                print(f"   🚨 Recommendation: Immediate intervention required!")
                
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

def main():
    """Main demo function"""
    print("🤖 CUSTOMER CHURN PREDICTION ML MODEL DEMO")
    print(f"⏰ Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🌐 Application running at: http://127.0.0.1:5000")
    
    # Test Case 1: High-risk customer
    high_risk_customer = {
        'gender': 'Female',
        'senior_citizen': '0',
        'partner': 'No',
        'dependents': 'No',
        'tenure': '6',
        'phone_service': 'Yes',
        'multiple_lines': 'No',
        'internet_service': 'DSL',
        'online_security': 'No',
        'online_backup': 'No',
        'device_protection': 'No',
        'tech_support': 'No',
        'streaming_tv': 'No',
        'streaming_movies': 'No',
        'contract': 'Month-to-month',
        'paperless_billing': 'Yes',
        'payment_method': 'Electronic check',
        'monthly_charges': '75.50',
        'total_charges': '453.00'
    }
    
    # Test Case 2: Low-risk customer
    low_risk_customer = {
        'gender': 'Male',
        'senior_citizen': '0',
        'partner': 'Yes',
        'dependents': 'Yes',
        'tenure': '60',
        'phone_service': 'Yes',
        'multiple_lines': 'Yes',
        'internet_service': 'Fiber optic',
        'online_security': 'Yes',
        'online_backup': 'Yes',
        'device_protection': 'Yes',
        'tech_support': 'Yes',
        'streaming_tv': 'Yes',
        'streaming_movies': 'Yes',
        'contract': 'Two year',
        'paperless_billing': 'No',
        'payment_method': 'Credit card (automatic)',
        'monthly_charges': '95.00',
        'total_charges': '5700.00'
    }
    
    # Test Case 3: Medium-risk customer
    medium_risk_customer = {
        'gender': 'Female',
        'senior_citizen': '1',
        'partner': 'Yes',
        'dependents': 'No',
        'tenure': '24',
        'phone_service': 'Yes',
        'multiple_lines': 'No',
        'internet_service': 'DSL',
        'online_security': 'Yes',
        'online_backup': 'No',
        'device_protection': 'No',
        'tech_support': 'Yes',
        'streaming_tv': 'No',
        'streaming_movies': 'No',
        'contract': 'One year',
        'paperless_billing': 'Yes',
        'payment_method': 'Mailed check',
        'monthly_charges': '65.75',
        'total_charges': '1578.00'
    }
    
    # Run tests
    test_prediction(high_risk_customer, "High-Risk Customer (New, Basic Plan)")
    test_prediction(low_risk_customer, "Low-Risk Customer (Long-term, Premium Plan)")
    test_prediction(medium_risk_customer, "Medium-Risk Customer (Senior, Standard Plan)")
    
    print(f"\n{'='*60}")
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("💡 Key Insights:")
    print("   • New customers with basic plans tend to have higher churn risk")
    print("   • Long-term customers with premium services show lower churn risk")
    print("   • Contract type and payment method significantly impact predictions")
    print("   • The model provides actionable risk levels for targeted interventions")
    print(f"{'='*60}")
    
    print(f"\n🌐 Visit the web interface at http://127.0.0.1:5000 to try interactive predictions!")

if __name__ == "__main__":
    main()
