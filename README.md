# Data Science Portfolio: Medical Analytics & AI

**Comprehensive portfolio demonstrating end-to-end data science capabilities in healthcare analytics**

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![R](https://img.shields.io/badge/R-4.3+-red)](https://r-project.org)
[![Azure](https://img.shields.io/badge/Azure-Cloud-blue)](https://azure.microsoft.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://docker.com)

## 🎯 Portfolio Overview

This portfolio showcases advanced data science skills through three interconnected projects analyzing fracture risk in elderly patients. Each project demonstrates different technical capabilities while maintaining data consistency and professional standards.

### 🚀 Live Demos
- **[ML API Dashboard](link-to-deployed-api)** - Interactive fracture risk prediction
- **[Patient Analytics Dashboard](link-to-shiny-app)** - Real-time patient monitoring
- **[Clinical NLP Assistant](link-to-azure-app)** - AI-powered clinical note analysis

## 📊 Projects

### 1. Fracture Risk ML Pipeline
**Technologies:** Python, Scikit-learn, FastAPI, Docker, MLflow

Complete machine learning pipeline for predicting fracture risk in elderly patients.

**Key Features:**
- Advanced feature engineering with medical domain knowledge
- Multiple ML algorithms (Random Forest, XGBoost, Neural Networks)
- Hyperparameter optimization with Optuna
- Model versioning and experiment tracking
- RESTful API with FastAPI
- Comprehensive testing and CI/CD

**Skills Demonstrated:**
- End-to-end ML pipeline development
- Model deployment and API development
- MLOps best practices
- Healthcare data analysis

### 2. Patient Dashboard (R Shiny)
**Technologies:** R, Shiny, plotly, DT, shinydashboard

Interactive web application for real-time patient monitoring and analytics.

**Key Features:**
- Real-time patient risk stratification
- Interactive visualizations with plotly
- Responsive design with custom CSS
- Data filtering and export capabilities
- Performance optimizations for large datasets

**Skills Demonstrated:**
- Full-stack web development in R
- Data visualization and UI/UX design
- Real-time data processing
- Healthcare dashboard development

### 3. Clinical NLP with Azure AI
**Technologies:** Python, Azure OpenAI, Azure Cognitive Services, Streamlit

AI-powered clinical note analysis using modern LLM capabilities.

**Key Features:**
- Clinical note summarization and entity extraction
- Risk factor identification from unstructured text
- Azure OpenAI integration for advanced NLP
- Streamlit web interface
- HIPAA-compliant data processing

**Skills Demonstrated:**
- Modern AI/LLM integration
- Cloud services (Azure)
- Natural language processing
- Healthcare text analytics

## 📈 Technical Highlights

### Data Engineering
- **Synthetic Data Generation**: Realistic medical datasets with proper statistical distributions
- **Data Validation**: Comprehensive data quality checks and validation pipelines
- **Feature Engineering**: Domain-specific feature creation for healthcare analytics

### Machine Learning
- **Model Development**: Multiple algorithms with proper validation and testing
- **Performance Optimization**: Hyperparameter tuning and model selection
- **Deployment**: Containerized APIs with monitoring and logging

### Software Engineering
- **Clean Code**: PEP 8 compliant Python, tidy R code with proper documentation
- **Testing**: Unit tests, integration tests, and data validation tests
- **CI/CD**: Automated testing and deployment pipelines
- **Documentation**: Comprehensive project documentation and API specs

## 🔬 Dataset Overview

All projects use a consistent synthetic medical dataset containing:

- **5,000 patients** with realistic demographic distributions
- **Clinical measurements** including bone density, lab results, and physical assessments
- **Fracture events** with detailed outcome tracking
- **Patient surveys** capturing quality of life and functional status
- **Clinical notes** for unstructured text analysis

**Key Dataset Features:**
- Realistic statistical distributions based on medical literature
- Proper handling of missing data and edge cases
- HIPAA-compliant synthetic data (no real patient information)
- Comprehensive data dictionary and documentation

## 🛠 Setup and Installation

### Prerequisites
- Python 3.9+
- R 4.3+
- Docker
- Git

### Quick Start
```bash
# Clone repository
git clone https://github.com/DawitLam/data-science-portfolio.git
cd data-science-portfolio

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Generate synthetic datasets
python shared/data_generators/synthetic_medical_data_generator.py

# Run individual projects (see project-specific READMEs)
```

### Environment Setup
```bash
# Using conda
conda env create -f environment.yml
conda activate data-portfolio

# Using Docker
docker-compose up -d
```

## 📁 Repository Structure

```
data-science-portfolio/
├── projects/
│   ├── 01-fracture-risk-ml/     # Python ML pipeline
│   ├── 02-patient-dashboard/    # R Shiny web app
│   └── 03-medical-nlp-azure/    # AI/NLP with Azure
├── data/
│   ├── synthetic/               # Generated datasets
│   └── processed/              # Feature-engineered data
├── shared/
│   ├── data_generators/        # Synthetic data scripts
│   ├── utils/                  # Common utilities
│   └── configs/               # Shared configurations
└── docs/                      # Portfolio documentation
```

## 🎯 Skills Demonstrated

### Data Science & Analytics
- **Statistical Analysis**: Hypothesis testing, survival analysis, risk modeling
- **Machine Learning**: Classification, regression, ensemble methods, deep learning
- **Data Visualization**: Interactive dashboards, publication-quality plots
- **Feature Engineering**: Domain-specific feature creation and selection

### Software Engineering
- **Python Development**: Object-oriented programming, API development, testing
- **R Development**: Shiny applications, package development, statistical computing
- **Web Development**: Full-stack applications, responsive design, user experience
- **DevOps**: Docker containerization, CI/CD pipelines, cloud deployment

### Healthcare Domain
- **Medical Data Analysis**: Understanding of clinical measurements and outcomes
- **Healthcare Compliance**: HIPAA-aware data handling and privacy protection
- **Clinical Workflow**: Integration with healthcare systems and processes
- **Evidence-Based Analysis**: Medical literature-informed feature engineering

## 🏆 Professional Impact

This portfolio demonstrates:
- **Production-Ready Code**: All projects include proper testing, documentation, and deployment
- **Domain Expertise**: Deep understanding of healthcare analytics and clinical workflows
- **Technical Versatility**: Proficiency across multiple languages, frameworks, and platforms
- **Business Acumen**: Focus on actionable insights and practical implementation

## 📞 Contact Information

**Dawit Lambebo Gulta**
- 📧 Email: dawit.lambebo@gmail.com
- 💼 LinkedIn: [linkedin.com/in/dawit-lambebo-gulta](https://www.linkedin.com/in/dawit-lambebo-gulta/)
- 🐙 GitHub: [github.com/DawitLam](https://github.com/DawitLam)
- 📍 Location: Toronto, Canada

---

## 🚀 Next Steps for Recruiters

1. **Explore the code**: Each project includes detailed documentation and clean, commented code
2. **Try the demos**: Live applications demonstrate real-world functionality
3. **Review the tests**: Comprehensive testing shows software engineering best practices
4. **Check the documentation**: Technical architecture and design decisions are well-documented

*This portfolio represents production-quality code and demonstrates readiness for senior data science and ML engineering roles.*
