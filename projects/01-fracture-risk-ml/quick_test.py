#!/usr/bin/env python3
"""
Quick API Test - Simple demonstration
"""

import requests
import json

def test_api():
    print("🚀 Fracture Risk Prediction API - Quick Demo")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1️⃣ Health Check:")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 2: Model info
    print("\n2️⃣ Model Information:")
    try:
        response = requests.get("http://localhost:8000/model/info", timeout=5)
        print(f"   Status: {response.status_code}")
        info = response.json()
        print(f"   Model: {info.get('model_type', 'N/A')}")
        print(f"   Features: {info.get('n_features', 'N/A')}")
        print(f"   Performance: {info.get('performance', {}).get('cv_auc_mean', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 3: High-risk patient prediction
    print("\n3️⃣ High-Risk Patient Prediction:")
    patient_high_risk = {
        "age": 75,
        "gender": "female",
        "weight": 50.0,
        "height": 155.0,
        "previous_fracture": True,
        "parent_fractured_hip": True,
        "current_smoking": True,
        "glucocorticoids": True,
        "rheumatoid_arthritis": False,
        "secondary_osteoporosis": True,
        "alcohol_3_or_more_units": False,
        "femoral_neck_bmd": 0.55
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=patient_high_risk,
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Risk Score: {result.get('risk_score', 0):.1%}")
        print(f"   Risk Category: {result.get('risk_category', 'N/A')}")
        print(f"   Recommendations: {len(result.get('clinical_recommendations', []))} provided")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 4: Low-risk patient prediction
    print("\n4️⃣ Low-Risk Patient Prediction:")
    patient_low_risk = {
        "age": 30,
        "gender": "male",
        "weight": 75.0,
        "height": 180.0,
        "previous_fracture": False,
        "parent_fractured_hip": False,
        "current_smoking": False,
        "glucocorticoids": False,
        "rheumatoid_arthritis": False,
        "secondary_osteoporosis": False,
        "alcohol_3_or_more_units": False,
        "femoral_neck_bmd": 1.0
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=patient_low_risk,
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Risk Score: {result.get('risk_score', 0):.1%}")
        print(f"   Risk Category: {result.get('risk_category', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    print("\n🎉 API Demo Complete!")
    print("📖 Visit http://localhost:8000/docs for interactive documentation")

if __name__ == "__main__":
    test_api()
