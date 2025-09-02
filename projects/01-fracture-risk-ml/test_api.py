#!/usr/bin/env python3
"""
Test script for the Fracture Risk Prediction API.

This script demonstrates how to use the API to predict fracture risk
for different patient scenarios.
"""

import requests
import json
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_endpoint() -> Dict[str, Any]:
    """Test the health endpoint to ensure API is running."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to API: {e}"}

def predict_fracture_risk(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Send a prediction request to the API."""
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=patient_data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Prediction failed: {e}"}

def main():
    """Run API tests with different patient scenarios."""
    
    print("🔬 Testing Fracture Risk Prediction API")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1️⃣ Testing Health Endpoint...")
    health = test_health_endpoint()
    if "error" in health:
        print(f"❌ {health['error']}")
        return
    else:
        print(f"✅ API Status: {health['status']}")
        print(f"✅ Model Status: {health['model_status']}")
    
    # Test 2: High-Risk Patient (elderly female with osteoporosis)
    print("\n2️⃣ Testing High-Risk Patient...")
    high_risk_patient = {
        "age": 75,
        "gender": "Female",
        "bmi": 19.5,  # Low BMI
        "spine_t_score": -3.2,  # Osteoporosis
        "hip_t_score": -2.8,  # Osteoporosis
        "vitamin_d_ng_ml": 15.0,  # Deficient
        "calcium_mg_dl": 8.8,
        "phosphorus_mg_dl": 3.2,
        "pth_pg_ml": 55.0,
        "grip_strength_kg": 15.0,  # Low
        "smoking_status": "Former",
        "alcohol_units_per_week": 5,
        "previous_fracture_count": 2,  # Multiple previous fractures
        "family_history_fractures": True,
        "exercise_frequency": "Rarely"
    }
    
    result = predict_fracture_risk(high_risk_patient)
    if "error" in result:
        print(f"❌ {result['error']}")
    else:
        print(f"🔴 Risk Score: {result['fracture_risk_score']:.4f}")
        print(f"🔴 Risk Category: {result['risk_category']}")
        print(f"🔴 Risk Factors: {', '.join(result['risk_factors'][:3])}")
        print(f"🔴 Top Recommendation: {result['recommendations'][0]}")
    
    # Test 3: Low-Risk Patient (young healthy male)
    print("\n3️⃣ Testing Low-Risk Patient...")
    low_risk_patient = {
        "age": 35,
        "gender": "Male", 
        "bmi": 24.0,  # Normal BMI
        "spine_t_score": 0.5,  # Normal
        "hip_t_score": 0.2,  # Normal
        "vitamin_d_ng_ml": 35.0,  # Adequate
        "calcium_mg_dl": 9.5,
        "phosphorus_mg_dl": 3.5,
        "pth_pg_ml": 30.0,
        "grip_strength_kg": 45.0,  # Strong
        "smoking_status": "Never",
        "alcohol_units_per_week": 3,
        "previous_fracture_count": 0,
        "family_history_fractures": False,
        "exercise_frequency": "Daily"
    }
    
    result = predict_fracture_risk(low_risk_patient)
    if "error" in result:
        print(f"❌ {result['error']}")
    else:
        print(f"🟢 Risk Score: {result['fracture_risk_score']:.4f}")
        print(f"🟢 Risk Category: {result['risk_category']}")
        print(f"🟢 Risk Factors: {', '.join(result['risk_factors']) if result['risk_factors'] else 'None identified'}")
        print(f"🟢 Top Recommendation: {result['recommendations'][0]}")
    
    # Test 4: Moderate-Risk Patient (middle-aged with osteopenia)
    print("\n4️⃣ Testing Moderate-Risk Patient...")
    moderate_risk_patient = {
        "age": 62,
        "gender": "Female",
        "bmi": 22.8,
        "spine_t_score": -1.8,  # Osteopenia
        "hip_t_score": -1.5,  # Osteopenia
        "vitamin_d_ng_ml": 25.0,  # Borderline
        "previous_fracture_count": 0,
        "family_history_fractures": True,
        "smoking_status": "Never",
        "exercise_frequency": "Weekly"
    }
    
    result = predict_fracture_risk(moderate_risk_patient)
    if "error" in result:
        print(f"❌ {result['error']}")
    else:
        print(f"🟡 Risk Score: {result['fracture_risk_score']:.4f}")
        print(f"🟡 Risk Category: {result['risk_category']}")
        print(f"🟡 Risk Factors: {', '.join(result['risk_factors'][:2])}")
        print(f"🟡 Top Recommendation: {result['recommendations'][0]}")
    
    print("\n" + "=" * 50)
    print("✅ API Testing Complete!")
    print("🌐 View full API documentation at: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
