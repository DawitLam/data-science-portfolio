"""
🤖 Patient Lookup Demo Script
============================

Use these patient IDs to test the enhanced ML prediction features:

HIGH RISK PATIENTS (should show 🔴 High Risk):
- FRAC_000001
- FRAC_000100  
- FRAC_000250
- FRAC_000500

MODERATE RISK PATIENTS (should show 🟡 Moderate Risk):
- FRAC_001000
- FRAC_001500
- FRAC_002000

LOW RISK PATIENTS (should show 🟢 Low Risk):
- FRAC_003000
- FRAC_004000
- FRAC_004500

DEMO SEQUENCE:
1. Navigate to "Patient Lookup" tab
2. Select model: C:\Users\Dama\...\models\best_model.pkl
3. Enter patient ID: FRAC_000001
4. Observe:
   - Patient demographics display
   - Risk probability calculation
   - Color-coded risk level
   - Top 5 risk factors
   - Clinical recommendations

EXPECTED RESULTS:
- Risk probability as percentage
- Risk level with appropriate color
- Feature importance ranking
- Personalized care recommendations

PORTFOLIO IMPACT:
- Demonstrates ML integration
- Shows clinical decision support
- Highlights healthcare domain knowledge
- Professional healthcare UI/UX
"""

# Sample patients to highlight in demo
DEMO_PATIENTS = {
    "high_risk": ["FRAC_000001", "FRAC_000100", "FRAC_000250"], 
    "moderate_risk": ["FRAC_001000", "FRAC_001500", "FRAC_002000"],
    "low_risk": ["FRAC_003000", "FRAC_004000", "FRAC_004500"]
}

print("🏥 Enhanced Patient Dashboard Demo Ready!")
print("📊 All 4 tabs enhanced with professional visualizations")
print("🤖 ML prediction engine integrated")
print("🎯 Portfolio-ready healthcare analytics platform")
