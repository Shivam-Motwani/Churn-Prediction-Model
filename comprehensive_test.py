#!/usr/bin/env python3
"""
Comprehensive Test for Churn Prediction Form Issues
===================================================
This script tests both the form submission and total charges validation.
"""

import requests
import time
import sys

def wait_for_server(url="http://127.0.0.1:5000", timeout=30):
    """Wait for the Flask server to be ready"""
    print("Waiting for Flask server to be ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
            print(".", end="", flush=True)
    print(f"\n❌ Server not ready after {timeout} seconds")
    return False

def test_form_submission(description, form_data):
    """Test form submission with given data"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {description}")
    print('='*60)
    
    # Show key form data
    print(f"📋 Key Data:")
    print(f"   Tenure: {form_data.get('tenure')} months")
    print(f"   Monthly Charges: ${form_data.get('monthly_charges')}")
    print(f"   Total Charges: ${form_data.get('total_charges')}")
    print(f"   Contract: {form_data.get('contract')}")
    
    try:
        # Submit form
        response = requests.post('http://127.0.0.1:5000/predict', data=form_data, timeout=10)
        
        print(f"\n📡 Response:")
        print(f"   Status Code: {response.status_code}")
        print(f"   Content Length: {len(response.text)} characters")
        
        # Check what page we got back
        if response.status_code == 200:
            if 'Churn Prediction Result' in response.text:
                print("✅ SUCCESS: Got result page!")
                
                # Extract prediction details
                import re
                
                # Extract probability
                prob_matches = re.findall(r'(\d+\.\d+)%', response.text)
                if prob_matches:
                    print(f"   🎯 Churn Probability: {prob_matches[0]}%")
                
                # Extract risk level
                risk_match = re.search(r'badge.*?>(\\w+)\\s+RISK</span>', response.text)
                if risk_match:
                    print(f"   ⚠️  Risk Level: {risk_match.group(1)}")
                
                # Check for recommendations
                if 'Recommended Actions' in response.text:
                    print("   💡 Recommendations section found")
                
                return True
                
            elif 'An error occurred during prediction' in response.text:
                print("❌ ERROR: Error message found in response")
                return False
                
            elif 'Predict Customer Churn' in response.text:
                print("⚠️  WARNING: Still on form page - likely validation error")
                
                # Check for validation messages
                if 'is-invalid' in response.text:
                    print("   📝 Form validation errors detected")
                
                return False
            else:
                print("❓ UNKNOWN: Unexpected page returned")
                return False
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏱️  ERROR: Request timed out")
        return False
    except requests.exceptions.ConnectionError:
        print("🔌 ERROR: Connection failed")
        return False
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 COMPREHENSIVE CHURN PREDICTION FORM TEST")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Wait for server
    if not wait_for_server():
        print("❌ Cannot proceed without server")
        sys.exit(1)
    
    # Test Case 1: Normal customer (should work)
    normal_customer = {
        'gender': 'Female',
        'senior_citizen': '0',
        'partner': 'Yes',
        'dependents': 'No',
        'tenure': '24',
        'phone_service': 'Yes',
        'multiple_lines': 'No',
        'internet_service': 'DSL',
        'online_security': 'No',
        'online_backup': 'Yes',
        'device_protection': 'No',
        'tech_support': 'No',
        'streaming_tv': 'No',
        'streaming_movies': 'No',
        'contract': 'One year',
        'paperless_billing': 'Yes',
        'payment_method': 'Electronic check',
        'monthly_charges': '65.50',
        'total_charges': '1572.00'  # 24 months * $65.50 = reasonable
    }
    
    # Test Case 2: Customer with potentially incorrect total charges
    incorrect_charges_customer = {
        'gender': 'Male',
        'senior_citizen': '0',
        'partner': 'No',
        'dependents': 'No',
        'tenure': '12',
        'phone_service': 'Yes',
        'multiple_lines': 'No',
        'internet_service': 'Fiber optic',
        'online_security': 'No',
        'online_backup': 'No',
        'device_protection': 'No',
        'tech_support': 'No',
        'streaming_tv': 'No',
        'streaming_movies': 'No',
        'contract': 'Month-to-month',
        'paperless_billing': 'Yes',
        'payment_method': 'Electronic check',
        'monthly_charges': '75.00',
        'total_charges': '100.00'  # Way too low for 12 months at $75/month
    }
    
    # Test Case 3: High-value long-term customer
    premium_customer = {
        'gender': 'Female',
        'senior_citizen': '1',
        'partner': 'Yes',
        'dependents': 'Yes',
        'tenure': '48',
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
        'monthly_charges': '105.00',
        'total_charges': '5040.00'  # 48 months * $105 = correct
    }
    
    # Run tests
    results = []
    results.append(test_form_submission("Normal Customer", normal_customer))
    results.append(test_form_submission("Customer with Questionable Total Charges", incorrect_charges_customer))
    results.append(test_form_submission("Premium Long-term Customer", premium_customer))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"✅ Successful: {success_count}/{total_count}")
    print(f"❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 ALL TESTS PASSED!")
    elif success_count > 0:
        print("⚠️  SOME TESTS PASSED - Check failed cases above")
    else:
        print("💥 ALL TESTS FAILED - Check Flask app and form processing")
    
    print(f"\n💡 If you see 'Still on form page' errors, the form validation might be rejecting the data.")
    print(f"💡 Check the Flask app console for error messages.")
    print(f"💡 Visit http://127.0.0.1:5000/predict to test manually.")

if __name__ == "__main__":
    main()
