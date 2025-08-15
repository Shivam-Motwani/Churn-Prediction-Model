import requests

def test_form_submission():
    """Test form submission to check if results are showing correctly"""
    
    # Test data with corrected total charges
    form_data = {
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
        'total_charges': '100.00'  # This should trigger validation
    }
    
    try:
        print("Testing form submission...")
        response = requests.post('http://127.0.0.1:5000/predict', data=form_data)
        
        print(f'Status Code: {response.status_code}')
        
        if response.status_code == 200:
            # Check if we got the result page
            if 'Churn Prediction Result' in response.text:
                print('✅ SUCCESS: Result page loaded correctly')
                # Extract some key information
                if 'Churn Probability' in response.text:
                    print('✅ Prediction probability found in response')
                if 'Risk Level' in response.text:
                    print('✅ Risk level found in response')
            elif 'Predict Customer Churn' in response.text:
                print('⚠️  WARNING: Still on prediction form page - form may not have submitted correctly')
            else:
                print('❌ ERROR: Unknown page returned')
                print(f"Response length: {len(response.text)}")
        else:
            print(f'❌ ERROR: HTTP {response.status_code}')
            print(response.text[:500])
            
    except requests.exceptions.ConnectionError:
        print('❌ ERROR: Cannot connect to Flask app. Make sure it\'s running on http://127.0.0.1:5000')
    except Exception as e:
        print(f'❌ ERROR: {e}')

if __name__ == "__main__":
    test_form_submission()
