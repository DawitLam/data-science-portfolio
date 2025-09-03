# 🏥 Patient Dashboard - Interactive Medical Data Visualization

> **Live Demo**: Interactive Streamlit dashboard showcasing ML-driven healthcare analytics with synthetic medical data

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](http://localhost:8501)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)](https://r-project.org)

## 🎯 **Portfolio Highlights**

This dashboard demonstrates **end-to-end healthcare data science capabilities**:
- ✅ **Synthetic Data Generation** - Privacy-safe medical datasets
- ✅ **Feature Engineering** - 80+ medical risk factors  
- ✅ **ML Pipeline** - Multiple algorithms (RF, XGBoost, LightGBM)
- ✅ **Interactive Visualization** - Real-time dashboard with Streamlit
- ✅ **Clinical Decision Support** - Patient-level risk scoring

## 🎨 **Dashboard Features**

### 📊 **Multi-Tab Interface**
- **Overview**: Cohort demographics (5,000+ patients) with age distribution
- **Cardiovascular**: Risk factor analysis and high-risk identification  
- **Fracture**: Bone health events and risk patterns
- **Patient Lookup**: Individual risk profiles with ML predictions

### 🤖 **ML Integration** 
- Real-time predictions using trained models
- Multiple model support (Random Forest, XGBoost, LightGBM)
- Automated feature preprocessing pipelines
- Clinical risk scoring with confidence intervals

## 🚀 **Quick Demo**

### **📱 Launch Dashboard**
```powershell
cd projects\02-patient-dashboard
streamlit run app.py --server.port 8501
```
**→ Open**: `http://localhost:8501`

### **🔍 Try These Features**
1. **Overview Tab**: See 5,000-patient cohort with age distribution
2. **Patient Lookup**: Search patient ID (e.g., "PAT_001") for ML predictions
3. **Cardiovascular Tab**: Explore risk factor distributions
4. **Model Selection**: Switch between different ML algorithms

## 📊 **Technical Implementation**

### **Data Pipeline**
```
Synthetic Generation → Feature Engineering → Model Training → Dashboard Deployment
```

### **Tech Stack**
- **Frontend**: Streamlit (Python), R Shiny (alternative)
- **ML**: scikit-learn, XGBoost, LightGBM
- **Data**: pandas, numpy, synthetic medical datasets
- **Visualization**: plotly, matplotlib, ggplot2 (R)

## 🏥 **Medical Domain Features**

- **Risk Stratification**: 10-year cardiovascular risk assessment
- **Feature Engineering**: 80+ clinical variables from 35 base features  
- **FRAX-Inspired**: Fracture risk assessment methodology
- **Population Health**: Cohort-level analytics and trends
- **Clinical Validation**: Synthetic data maintains medical realism

## 💼 **For Recruiters**

### **🎯 Demo Highlights**
- **Interactive**: Click through tabs, search patients, see real-time predictions
- **Technical Depth**: Full ML pipeline from data → model → deployment
- **Domain Expertise**: Healthcare-specific feature engineering and risk modeling
- **Production Ready**: Robust error handling, scalable architecture

### **📱 Quick Setup**
```powershell
# One command to see the full dashboard
streamlit run app.py --server.port 8501
```

### **🎨 Screenshot Gallery**
*Dashboard showing 5,000-patient cohort with age distribution and interactive ML predictions*

---

## 🛠 **Development Setup**

### **Streamlit (Python)**
```powershell
cd projects\02-patient-dashboard
streamlit run app.py
```

### **R Shiny (Alternative)**
```powershell
R -e "shiny::runApp('shiny/', launch.browser=TRUE)"
```

### **Data Sources**
- Auto-loads from `data/synthetic/`
- Generates missing datasets automatically
- 5,000+ synthetic patients with clinical realism

---

## 🎪 **Status Notes**

This is a **portfolio demonstration project** showcasing:
- ✅ **Working Features**: Overview, demographics, basic ML scoring
- 🚧 **In Development**: Advanced cardiovascular analytics, complete model integration
- 💡 **Future Enhancements**: Real-time model retraining, clinical alerts

*Some features may be partially implemented - this demonstrates real-world project development and prioritization skills.*
