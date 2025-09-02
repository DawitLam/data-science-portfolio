#!/usr/bin/env python3
"""
Simple API demonstration script
Shows the fracture risk prediction API in action
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("\n" + "="*50)
    print("🏥 Testing Health Check Endpoint")
    print("="*50)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_info_endpoint():
    """Test the model info endpoint"""
    print("\n" + "="*50)
    print("📊 Testing Model Info Endpoint")
    print("="*50)
    
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prediction_endpoint():
    """Test the prediction endpoint with example patient data"""
    print("\n" + "="*50)
    print("🔮 Testing Fracture Risk Prediction")
    print("="*50)
    
    # Example patient: 65-year-old female with several risk factors
    patient_data = {
        "age": 65,
        "gender": "female",
        "weight": 55.0,
        "height": 160.0,
        "previous_fracture": True,
        "parent_fractured_hip": True,
        "current_smoking": False,
        "glucocorticoids": True,
        "rheumatoid_arthritis": False,
        "secondary_osteoporosis": False,
        "alcohol_3_or_more_units": False,
        "femoral_neck_bmd": 0.65
    }
    
    print("👤 Patient Profile:")
    for key, value in patient_data.items():
        print(f"   {key}: {value}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=patient_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"\n📋 Prediction Results:")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Extract key information
            risk_score = result.get('risk_score', 0)
            risk_category = result.get('risk_category', 'Unknown')
            recommendations = result.get('clinical_recommendations', [])
            
            print(f"\n🎯 Key Results:")
            print(f"   Risk Score: {risk_score:.1%}")
            print(f"   Risk Category: {risk_category}")
            print(f"   Clinical Recommendations:")
            for rec in recommendations:
                print(f"     • {rec}")
        else:
            print(f"Error: {response.text}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_low_risk_patient():
    """Test with a low-risk patient"""
    print("\n" + "="*50)
    print("🟢 Testing Low-Risk Patient Prediction")
    print("="*50)
    
    # Example: Young, healthy individual
    patient_data = {
        "age": 35,
        "gender": "male",
        "weight": 75.0,
        "height": 175.0,
        "previous_fracture": False,
        "parent_fractured_hip": False,
        "current_smoking": False,
        "glucocorticoids": False,
        "rheumatoid_arthritis": False,
        "secondary_osteoporosis": False,
        "alcohol_3_or_more_units": False,
        "femoral_neck_bmd": 0.95
    }
    
    print("👤 Low-Risk Patient Profile:")
    for key, value in patient_data.items():
        print(f"   {key}: {value}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=patient_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            risk_score = result.get('risk_score', 0)
            risk_category = result.get('risk_category', 'Unknown')
            
            print(f"\n🎯 Results:")
            print(f"   Risk Score: {risk_score:.1%}")
            print(f"   Risk Category: {risk_category}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run the API demonstration"""
    print("🚀 Fracture Risk Prediction API Demo")
    print("====================================")
    print("This demonstration shows our ML-powered clinical decision support system")
    print("for fracture risk assessment, based on FRAX methodology.")
    
    # Wait a moment for API to be ready
    print("\n⏳ Waiting for API to be ready...")
    time.sleep(2)
    
    # Run tests
    tests = [
        ("Health Check", test_health_endpoint),
        ("Model Info", test_model_info_endpoint),
        ("High-Risk Prediction", test_prediction_endpoint),
        ("Low-Risk Prediction", test_low_risk_patient),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name} - PASSED")
        else:
            print(f"❌ {test_name} - FAILED")
    
    # Summary
    print("\n" + "="*50)
    print("📊 DEMONSTRATION SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The API is working perfectly.")
        print("\n🌐 You can also test the API interactively at:")
        print("   http://localhost:8000/docs")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the API server.")

if __name__ == "__main__":
    main()
