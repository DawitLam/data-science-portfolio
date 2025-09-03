# Cardiovascular Risk Project (lightweight demo)

This project demonstrates an end-to-end cardiovascular risk prediction pipeline using synthetic/demo data.

What it contains
- ETL: `src/etl.py` — reading, date parsing, computing age, basic normalization.
- Cleaning: `src/cleaning.py` — missing-value imputation and outlier removal.
- ML analysis: `src/ml_analysis.py` — small model suite (logistic, random forest, optional xgboost), evaluation, feature importance and interactive ROC helper.
- Pipeline runner: `run_pipeline.py` — lightweight CLI that wires ETL→cleaning→train→save.

Quick-start (PowerShell)

1. Activate your virtualenv (example path used in repository):

```powershell
& "C:/Users/Dama/Documents/Python project/portfolio/data-science-portfolio/.venv/Scripts/Activate.ps1"
```

2. From the repo root run the pipeline (example target column name: `target`):

```powershell
cd projects\02-cardiovascular-risk-ml
& "C:/Users/Dama/Documents/Python project/portfolio/data-science-portfolio/.venv/Scripts/python.exe" run_pipeline.py --target target
```

3. Artifacts (models and `training_results_*.json`) are saved under `projects/02-cardiovascular-risk-ml/models/`.

If you don't have large source data, the pipeline will use `demo_small.csv` at the repository root if present. This keeps the demo lightweight for recruiters.

Next steps
- Add a small notebook cell to load `models/best_model.pkl` and show the interactive ROC.
- Add unit tests for `etl.basic_etl` and `cleaning.impute_missing`.
# 🫀 Cardiovascular Disease Risk Assessment ML Pipeline

## Project Overview

A comprehensive machine learning system for predicting cardiovascular disease risk using synthetic patient data. This project demonstrates end-to-end ML pipeline development with clinical applications, featuring advanced feature engineering, multiple model comparison, and a production-ready API.

## 🎯 Objectives

- **Primary Goal**: Develop a robust ML model to predict 10-year cardiovascular disease risk
- **Clinical Application**: Support preventive cardiology and risk stratification
- **Technical Demonstration**: Showcase advanced ML techniques and software engineering practices

## 🏥 Medical Context

Cardiovascular disease (CVD) is the leading cause of death globally. Early risk assessment enables:
- Preventive interventions
- Lifestyle modifications
- Targeted screening programs
- Resource allocation optimization

## 📊 Features & Risk Factors

### Primary Risk Factors
- **Demographics**: Age, gender, ethnicity
- **Clinical Measures**: Blood pressure, cholesterol levels, BMI
- **Lifestyle Factors**: Smoking status, physical activity, diet quality
- **Medical History**: Diabetes, family history, previous CVD events
- **Laboratory Values**: Glucose, inflammatory markers, lipid profiles

### Engineered Features
- **Composite Risk Scores**: Framingham-inspired calculations
- **Interaction Terms**: Age × gender, smoking × cholesterol
- **Normalized Metrics**: BMI categories, blood pressure stages
- **Temporal Features**: Risk factor progression indicators

## 🔧 Technical Architecture

### ML Pipeline Components
1. **Data Generation**: Synthetic cardiovascular dataset creation
2. **Feature Engineering**: Clinical domain-specific transformations
3. **Model Training**: Multiple algorithm comparison and selection
4. **Model Evaluation**: Medical-relevant metrics and validation
5. **API Deployment**: FastAPI server for real-time predictions

### Model Performance
- **Target Metric**: AUC-ROC (Area Under Receiver Operating Characteristic)
- **Clinical Metric**: Sensitivity/Specificity balance for screening
- **Validation**: K-fold cross-validation with temporal considerations

## 🚀 Quick Start

```bash
# Navigate to project directory
cd projects/02-cardiovascular-risk-ml

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python src/data/generate_data.py

# Train models
python src/models/train_model.py

# Start API server
python start_api.py
```

### Try it locally (Windows PowerShell)
Use these commands in PowerShell to run a quick local demo. These assume you have a repo-level virtual environment as in the main README.

```powershell
# From repo root
Set-Location .\projects\02-cardiovascular-risk-ml

# Activate repo virtual environment
.\.venv\Scripts\Activate.ps1

# Generate synthetic data
python .\src\data\generate_data.py

# Train models
python .\src\models\train_model.py

# Start API server
python .\start_api.py
```

## 📁 Project Structure

```
02-cardiovascular-risk-ml/
├── config/
│   ├── model_config.yaml      # Model hyperparameters
│   ├── feature_config.yaml    # Feature engineering settings
│   └── api_config.yaml        # API configuration
├── src/
│   ├── data/
│   │   ├── load_data.py       # Data loading and validation
│   │   └── generate_data.py   # Synthetic data generation
│   ├── features/
│   │   └── build_features.py  # Feature engineering pipeline
│   ├── models/
│   │   ├── train_model.py     # Model training and selection
│   │   └── predict_model.py   # Prediction utilities
│   └── api/
│       └── main.py            # FastAPI application
├── models/
│   └── trained_models/        # Saved model artifacts
├── notebooks/
│   └── cardiovascular_risk_analysis.ipynb  # EDA and prototyping
└── tests/
    └── test_pipeline.py       # Unit tests
```

## 🔬 Models Implemented

1. **Logistic Regression**: Baseline interpretable model
2. **Random Forest**: Ensemble method with feature importance
3. **XGBoost**: Gradient boosting for high performance
4. **LightGBM**: Efficient gradient boosting variant
5. **Neural Network**: Deep learning approach (optional)

## 📈 Key Features

### Clinical Validation
- **Risk Stratification**: Low, moderate, high risk categories
- **Interpretable Predictions**: Feature importance and SHAP values
- **Clinical Guidelines**: Alignment with ACC/AHA guidelines
- **Bias Detection**: Fairness across demographic groups

### Production Readiness
- **API Endpoints**: RESTful service for real-time predictions
- **Input Validation**: Pydantic models for data validation
- **Error Handling**: Comprehensive error management
- **Documentation**: Interactive API docs with Swagger/OpenAPI

## 🛡️ Privacy & Ethics

- **Synthetic Data Only**: No real patient information used
- **Bias Mitigation**: Regular fairness audits across demographics
- **Transparency**: Clear model explanations and limitations
- **HIPAA Considerations**: Privacy-by-design architecture

## 📋 Requirements

### Core Dependencies
- Python 3.8+
- pandas, numpy, scikit-learn
- xgboost, lightgbm
- fastapi, uvicorn
- pydantic, pyyaml

### Development Tools
- pytest for testing
- black for code formatting
- mypy for type checking
- pre-commit for code quality

## 🎓 Learning Outcomes

This project demonstrates:
- **Medical ML Applications**: Healthcare-specific modeling considerations
- **Feature Engineering**: Domain expertise in cardiovascular risk factors
- **Model Selection**: Systematic comparison of ML algorithms
- **Production Deployment**: API development and deployment practices
- **Clinical Validation**: Medical accuracy and interpretability

## 📚 References

- Framingham Heart Study Risk Algorithms
- ACC/AHA Cardiovascular Risk Guidelines
- WHO Cardiovascular Disease Prevention Guidelines
- SCORE2 European Risk Assessment System

---

**Note**: This project uses entirely synthetic data for educational and portfolio purposes. All clinical relationships are approximated based on published medical literature.
