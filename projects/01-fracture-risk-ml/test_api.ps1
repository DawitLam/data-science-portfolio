# Simple PowerShell test for the Fracture Risk API

Write-Host "🔬 Testing Fracture Risk Prediction API" -ForegroundColor Green
Write-Host "=" -repeat 50

# Test health endpoint
Write-Host "`n1️⃣ Testing Health Endpoint..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
    Write-Host "✅ API Status: $($health.status)" -ForegroundColor Green
    Write-Host "✅ Model Status: $($health.model_status)" -ForegroundColor Green
} catch {
    Write-Host "❌ Health check failed: $($_.Exception.Message)" -ForegroundColor Red
    exit
}

# Test prediction with high-risk patient
Write-Host "`n2️⃣ Testing High-Risk Patient Prediction..." -ForegroundColor Yellow

$highRiskPatient = @{
    age = 75
    gender = "Female"
    bmi = 19.5
    spine_t_score = -3.2
    hip_t_score = -2.8
    vitamin_d_ng_ml = 15.0
    previous_fracture_count = 2
    family_history_fractures = $true
    smoking_status = "Former"
    exercise_frequency = "Rarely"
} | ConvertTo-Json

try {
    $prediction = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $highRiskPatient -ContentType "application/json"
    Write-Host "🔴 Risk Score: $($prediction.fracture_risk_score)" -ForegroundColor Red
    Write-Host "🔴 Risk Category: $($prediction.risk_category)" -ForegroundColor Red
    Write-Host "🔴 Risk Factors: $($prediction.risk_factors -join ', ')" -ForegroundColor Red
    Write-Host "🔴 Top Recommendation: $($prediction.recommendations[0])" -ForegroundColor Red
} catch {
    Write-Host "❌ Prediction failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n" + "=" * 50
Write-Host "✅ API Test Complete!" -ForegroundColor Green
Write-Host "🌐 View full documentation at: http://localhost:8000/docs" -ForegroundColor Cyan
