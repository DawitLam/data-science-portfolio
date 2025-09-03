# Fracture Risk ML Pipeline

**Advanced machine learning pipeline for predicting fracture risk in elderly patients**

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)

## 🎯 Project Overview

This project demonstrates a complete end-to-end machine learning pipeline for predicting fracture risk in elderly patients. It showcases advanced ML engineering practices, healthcare domain expertise, and production-ready deployment capabilities.

### 🚀 Key Features

- **Advanced Feature Engineering**: Medical domain-specific features and transformations
- **Multiple ML Algorithms**: Random Forest, XGBoost, LightGBM, and Neural Networks
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Model Interpretability**: SHAP values and feature importance analysis
- **RESTful API**: FastAPI-based model serving
- **MLOps Integration**: Model versioning with MLflow
- **Comprehensive Testing**: Unit tests and integration tests
- **Docker Deployment**: Containerized application

## 📊 Model Performance

Current best model performance on validation set:
- **Algorithm**: XGBoost Classifier
- **AUC-ROC**: 0.847
- **Precision**: 0.782
- **Recall**: 0.734
- **F1-Score**: 0.757

## 🗂️ Project Structure

```
01-fracture-risk-ml/
├── config/
│   ├── model_config.yaml       # Model hyperparameters
│   ├── feature_config.yaml     # Feature engineering settings
│   └── api_config.yaml         # API configuration
├── data/
│   ├── raw/                    # Raw data (gitignored)
│   ├── processed/              # Processed datasets
│   └── features/               # Feature-engineered data
├── models/
│   ├── trained_models/         # Serialized models
│   ├── experiments/            # MLflow experiment tracking
│   └── artifacts/              # Model artifacts and metrics
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_model_interpretation.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py        # Data loading utilities
│   │   ├── preprocess.py       # Data preprocessing
│   │   └── validate.py         # Data validation
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py   # Feature engineering
│   │   ├── medical_features.py # Domain-specific features
│   │   └── selection.py        # Feature selection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py      # Model training pipeline
│   │   ├── predict_model.py    # Prediction pipeline
│   │   ├── evaluate.py         # Model evaluation
│   │   └── optimize.py         # Hyperparameter optimization
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI application
│   │   ├── models.py           # Pydantic models
│   │   └── endpoints.py        # API endpoints
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Configuration management
│       ├── logging.py          # Logging utilities
│       └── metrics.py          # Custom metrics
├── tests/
│   ├── __init__.py
│   ├── test_data/              # Test data processing
│   ├── test_features/          # Test feature engineering
│   ├── test_models/            # Test model training
│   └── test_api/               # Test API endpoints
├── deployment/
│   ├── Dockerfile              # Docker configuration
│   ├── docker-compose.yml      # Multi-container setup
│   ├── requirements.txt        # Python dependencies
│   └── startup.sh              # Application startup script
├── docs/
│   ├── api_documentation.md    # API documentation
│   ├── model_documentation.md  # Model methodology
│   └── deployment_guide.md     # Deployment instructions
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/DawitLam/data-science-portfolio.git
cd data-science-portfolio/projects/01-fracture-risk-ml
```

2. **Set up virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Generate synthetic data** (from project root)
```bash
cd ../..
python shared/data_generators/synthetic_medical_data_generator.py
cd projects/01-fracture-risk-ml
```

4. **Run the training pipeline**
```bash
python src/models/train_model.py
```

### Try it locally (Windows PowerShell)
Copy these commands into PowerShell. They assume you've already created the repo-level virtual environment as shown in the main README.

```powershell
# From repo root
Set-Location .\projects\01-fracture-risk-ml

# Activate repo venv (if created at repo root)
.\.venv\Scripts\Activate.ps1

# Train a model (quick demo)
python .\src\models\train_model.py

# Start the API (open http://127.0.0.1:8000/docs)
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

5. **Start the API server**
```bash
uvicorn src.api.main:app --reload
```

## 🔬 Usage Examples

### Training a Model
```python
from src.models.train_model import train_fracture_risk_model
from src.data.load_data import load_training_data

# Load data
X_train, X_val, y_train, y_val = load_training_data()

# Train model
model, metrics = train_fracture_risk_model(
    X_train, y_train, 
    X_val, y_val,
    algorithm='xgboost'
)

print(f"Model AUC: {metrics['auc']:.3f}")
```

### Making Predictions
```python
from src.models.predict_model import FractureRiskPredictor

# Initialize predictor
predictor = FractureRiskPredictor('models/trained_models/best_model.pkl')

# Make prediction
patient_data = {
    'age': 72,
    'gender': 'Female',
    'bmi': 22.1,
    'spine_t_score': -2.8,
    'hip_t_score': -2.1,
    'vitamin_d_ng_ml': 18.5,
    'grip_strength_kg': 18.2,
    'previous_fracture_count': 1
}

risk_score = predictor.predict_risk(patient_data)
print(f"Fracture Risk Score: {risk_score:.3f}")
```

### API Usage
```bash
# Health check
curl http://localhost:8000/health

# Predict fracture risk
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 72,
    "gender": "Female",
    "bmi": 22.1,
    "spine_t_score": -2.8,
    "hip_t_score": -2.1,
    "vitamin_d_ng_ml": 18.5,
    "grip_strength_kg": 18.2,
    "previous_fracture_count": 1
  }'
```

## 📈 Model Development Process

### 1. Data Exploration & Analysis
- **Statistical Analysis**: Distribution analysis, correlation studies
- **Medical Validation**: Clinical plausibility checks
- **Missing Data Analysis**: Patterns and imputation strategies

### 2. Feature Engineering
- **Medical Domain Features**: FRAX score components, risk ratios
- **Derived Variables**: BMI categories, T-score classifications
- **Interaction Features**: Age-gender, BMI-bone density interactions
- **Temporal Features**: Time since measurements, seasonal effects

### 3. Model Selection & Training
- **Baseline Models**: Logistic Regression, Random Forest
- **Advanced Models**: XGBoost, LightGBM, CatBoost
- **Ensemble Methods**: Voting classifiers, stacking
- **Deep Learning**: Neural networks for complex patterns

### 4. Model Optimization
- **Hyperparameter Tuning**: Optuna-based optimization
- **Cross-Validation**: Stratified K-fold validation
- **Feature Selection**: Recursive feature elimination, SHAP-based selection
- **Class Imbalance**: SMOTE, class weights, threshold optimization

### 5. Evaluation & Interpretation
- **Performance Metrics**: AUC-ROC, Precision-Recall, Calibration
- **Clinical Metrics**: Sensitivity, Specificity, PPV, NPV
- **Model Interpretation**: SHAP values, LIME, feature importance
- **Bias Analysis**: Fairness across demographic groups

## 🏥 Clinical Context

### Fracture Risk Factors
The model incorporates evidence-based risk factors from medical literature:

**Primary Risk Factors**:
- Age (exponential increase after 65)
- Gender (postmenopausal women highest risk)
- Bone mineral density (T-scores)
- Previous fracture history
- Family history of fractures

**Secondary Risk Factors**:
- Low BMI (<20 kg/m²)
- Smoking status
- Alcohol consumption
- Physical activity level
- Vitamin D deficiency
- Certain medications

### Model Validation
- **Clinical Validation**: Compared against FRAX calculator
- **External Validation**: Performance on holdout populations
- **Temporal Validation**: Performance over time periods
- **Subgroup Analysis**: Performance across age/gender groups

## 🔧 Configuration

### Model Configuration (`config/model_config.yaml`)
```yaml
models:
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42
  
  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
    min_samples_leaf: 2
    random_state: 42

training:
  test_size: 0.2
  validation_size: 0.2
  cross_validation_folds: 5
  random_state: 42

optimization:
  n_trials: 100
  timeout: 3600  # 1 hour
  direction: maximize
  metric: roc_auc
```

## 🐳 Docker Deployment

### Build and Run
```bash
# Build Docker image
docker build -t fracture-risk-api .

# Run container
docker run -p 8000:8000 fracture-risk-api

# Or use docker-compose
docker-compose up
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/test_models/
```

### Test Categories
- **Unit Tests**: Individual function testing
- **Integration Tests**: End-to-end pipeline testing
- **API Tests**: Endpoint functionality testing
- **Data Tests**: Data validation and schema testing

## 📊 Monitoring & MLOps

### MLflow Integration
- **Experiment Tracking**: Model parameters, metrics, artifacts
- **Model Registry**: Version control and stage management
- **Model Serving**: Direct model deployment capabilities

### Monitoring Dashboard
- **Model Performance**: Real-time accuracy tracking
- **Data Drift**: Input distribution monitoring
- **Prediction Distribution**: Output pattern analysis
- **System Health**: API response times, error rates

## 🔮 Future Enhancements

### Technical Improvements
- **Real-time Predictions**: Streaming data processing
- **Federated Learning**: Multi-site model training
- **Automated Retraining**: Continuous model updates
- **A/B Testing**: Model variant comparison

### Clinical Extensions
- **Multi-outcome Prediction**: Hip vs. vertebral fractures
- **Treatment Recommendations**: Personalized interventions
- **Risk Trajectories**: Longitudinal risk modeling
- **Integration**: EHR system connectivity

## 📚 References

1. Kanis JA, et al. FRAX and the assessment of fracture probability in men and women from the UK. Osteoporos Int. 2008.
2. Cosman F, et al. Clinician's Guide to Prevention and Treatment of Osteoporosis. Osteoporos Int. 2014.
3. Compston J, et al. UK clinical guideline for the prevention and treatment of osteoporosis. Arch Osteoporos. 2017.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Dawit Lambebo Gulta**
- Email: dawit.lambebo@gmail.com
- LinkedIn: [linkedin.com/in/dawit-lambebo-gulta](https://www.linkedin.com/in/dawit-lambebo-gulta/)
- GitHub: [github.com/DawitLam](https://github.com/DawitLam)
