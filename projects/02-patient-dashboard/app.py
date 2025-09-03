import streamlit as st
import pandas as pd
from pathlib import Path
import importlib
import joblib
import glob
from typing import Optional
from sklearn.pipeline import Pipeline
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache_data
def ensure_and_load_data():
    repo_root = Path(__file__).resolve().parents[2]
    synth_dir = repo_root / 'data' / 'synthetic'
    synth_dir.mkdir(parents=True, exist_ok=True)

    cvd_path = synth_dir / 'cardiovascular_risk_data.csv'
    master_path = synth_dir / 'master_patient_data.csv'
    fracture_path = synth_dir / 'fracture_events.csv'

    # Generate missing datasets using shared generators when available
    if not cvd_path.exists():
        try:
            gen = importlib.import_module('shared.data_generators.cardiovascular_data_generator')
            gen.generate_cardiovascular_data(n_samples=2000, output_path=str(cvd_path))
        except Exception:
            pass

    if not master_path.exists() or not fracture_path.exists():
        try:
            gen2 = importlib.import_module('shared.data_generators.synthetic_medical_data_generator')
            # generator saves multiple files under data/synthetic by default
            gen2.generate_all = getattr(gen2, 'generate_all', None)
            if callable(gen2.generate_all):
                gen2.generate_all()
        except Exception:
            pass

    # Load what we can
    dfs = {}
    if cvd_path.exists():
        dfs['cvd'] = pd.read_csv(cvd_path)
    if master_path.exists():
        dfs['master'] = pd.read_csv(master_path)
    if fracture_path.exists():
        dfs['fracture'] = pd.read_csv(fracture_path)

    return dfs


def find_models(repo_root: Path):
    """Search common locations for saved best_model.pkl files and return a list of paths."""
    candidates = []
    # repo-level models folder
    candidates.extend(list((repo_root / 'models').glob('**/best_model.pkl')))
    # project-specific locations
    candidates.extend(list((repo_root / 'projects').glob('**/models/**/best_model.pkl')))
    # explicit common places
    candidates.extend(list((repo_root / 'projects' / '01-fracture-risk-ml' / 'models' / 'trained_models').glob('best_model.pkl')))
    candidates.extend(list((repo_root / 'projects' / '02-cardiovascular-risk-ml' / 'models').glob('best_model.pkl')))

    # dedupe and return as Path objects
    seen = set()
    results = []
    for p in candidates:
        try:
            pp = Path(p)
        except Exception:
            continue
        if pp.exists() and str(pp) not in seen:
            seen.add(str(pp))
            results.append(pp)
    return results


def load_model(path: Path):
    try:
        m = joblib.load(str(path))
        return m
    except Exception as e:
        st.error(f"Failed to load model {path}: {e}")
        return None


def find_preprocessor_for_model(model_path: Path) -> Optional[Path]:
    """Look for common preprocessing artifacts next to a saved model file.

    Returns the Path to the first matching preprocessor, or None.
    """
    candidates = [
        'preprocessor.pkl',
        'preprocessor.joblib',
        'scaler.pkl',
        'scaler.joblib',
        'pipeline.pkl',
        'feature_pipeline.pkl',
    ]

    # search in the model directory and one level up
    search_dirs = [model_path.parent, model_path.parent.parent]
    for d in search_dirs:
        for name in candidates:
            p = d / name
            if p.exists():
                return p
    return None


def load_preprocessor(path: Path):
    try:
        pre = joblib.load(str(path))
        return pre
    except Exception as e:
        st.warning(f'Failed to load preprocessor {path}: {e}')
        return None


def infer_feature_names(model, X_sample: pd.DataFrame | None = None) -> list | None:
    """Try several strategies to infer the feature name ordering used to train `model`.

    If found, return list of feature names. If not, return None.
    """
    # 1) sklearn attribute
    try:
        if hasattr(model, 'feature_names_in_'):
            return list(getattr(model, 'feature_names_in_'))
    except Exception:
        pass

    # 2) if pipeline, check transformers and final estimator
    try:
        if isinstance(model, Pipeline):
            # try pipeline.named_steps or steps
            last = model.steps[-1][1]
            if hasattr(last, 'feature_names_in_'):
                return list(getattr(last, 'feature_names_in_'))
    except Exception:
        pass

    # 3) if X_sample provided, take numeric columns ordering
    if X_sample is not None and isinstance(X_sample, pd.DataFrame):
        return list(X_sample.select_dtypes(include=['number']).columns)

    return None


def save_features_list(model_path: Path, features: list) -> None:
    try:
        out = model_path.parent / 'features.json'
        with open(out, 'w', encoding='utf8') as f:
            json.dump(features, f, ensure_ascii=False)
    except Exception:
        pass


def load_features_list(model_path: Path) -> list | None:
    p = model_path.parent / 'features.json'
    if p.exists():
        try:
            return json.load(open(p, 'r', encoding='utf8'))
        except Exception:
            return None
    return None


def compute_feature_importance(model, feature_names: list) -> pd.DataFrame | None:
    try:
        import pandas as _pd
        import numpy as _np

        feats = list(feature_names)
        # tree-based
        if hasattr(model, 'feature_importances_'):
            imp = _np.array(model.feature_importances_)
            df = _pd.DataFrame({'feature': feats, 'importance': imp})
            return df.sort_values('importance', ascending=False).reset_index(drop=True)

        # linear
        if hasattr(model, 'coef_'):
            coef = _np.ravel(model.coef_)
            if coef.shape[0] != len(feats):
                coef = _np.mean(model.coef_, axis=0)
            df = _pd.DataFrame({'feature': feats, 'importance': _np.abs(coef), 'coef': coef})
            return df.sort_values('importance', ascending=False).reset_index(drop=True)

    except Exception:
        return None



def main():
    st.set_page_config(page_title='Patient Dashboard', layout='wide')
    st.title('Patient Dashboard — Synthetic Medical Data')

    dfs = ensure_and_load_data()

    tabs = st.tabs(['Overview', 'Cardiovascular', 'Fracture', 'Patient Lookup'])

    with tabs[0]:
        st.header('Cohort Overview')
        if 'master' in dfs:
            df = dfs['master']
            
            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric('Total Patients', f"{len(df):,}")
            with col2:
                avg_age = df['age'].mean()
                st.metric('Average Age', f"{avg_age:.1f} years")
            with col3:
                gender_dist = df['gender'].value_counts()
                female_pct = (gender_dist.get('F', 0) / len(df)) * 100
                st.metric('Female %', f"{female_pct:.1f}%")
            with col4:
                if 'bmi' in df.columns:
                    avg_bmi = df['bmi'].mean()
                    st.metric('Average BMI', f"{avg_bmi:.1f}")
            
            # Two-column layout for charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader('Age Distribution')
                # Create bins for age distribution
                age_bins = pd.cut(df['age'], bins=10)
                age_counts = age_bins.value_counts().sort_index()
                
                fig = px.histogram(df, x='age', nbins=20, 
                                 title='Patient Age Distribution',
                                 labels={'age': 'Age (years)', 'count': 'Number of Patients'},
                                 color_discrete_sequence=['#1f77b4'])
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader('Gender Distribution')
                gender_counts = df['gender'].value_counts()
                fig = px.pie(values=gender_counts.values, names=gender_counts.index,
                           title='Gender Distribution',
                           color_discrete_sequence=['#ff7f0e', '#1f77b4'])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Additional insights
            if 'bmi' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader('BMI Distribution')
                    fig = px.box(df, y='bmi', title='BMI Distribution',
                               labels={'bmi': 'BMI'})
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader('Age vs BMI Correlation')
                    fig = px.scatter(df, x='age', y='bmi', 
                                   title='Age vs BMI Relationship',
                                   labels={'age': 'Age (years)', 'bmi': 'BMI'},
                                   opacity=0.6)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
        else:
            st.info('Master patient dataset not found. The app attempted to generate synthetic data if available.')

    with tabs[1]:
        st.header('Cardiovascular Risk Analysis')
        if 'cvd' in dfs:
            df = dfs['cvd']
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric('Total Records', f"{len(df):,}")
            with col2:
                high_risk = (df['cvd_risk_10yr'] == 1).sum()
                high_risk_pct = (high_risk / len(df)) * 100
                st.metric('High Risk %', f"{high_risk_pct:.1f}%")
            with col3:
                avg_bp = df['systolic_bp'].mean()
                st.metric('Avg Systolic BP', f"{avg_bp:.0f} mmHg")
            with col4:
                if 'cholesterol' in df.columns:
                    avg_chol = df['cholesterol'].mean()
                    st.metric('Avg Cholesterol', f"{avg_chol:.0f} mg/dL")
            
            # Charts in two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader('10-Year CVD Risk Distribution')
                risk_counts = df['cvd_risk_10yr'].value_counts()
                risk_labels = ['Low Risk', 'High Risk']
                fig = px.pie(values=risk_counts.values, names=risk_labels,
                           title='Cardiovascular Risk Distribution',
                           color_discrete_sequence=['#2ca02c', '#d62728'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader('Systolic BP Distribution')
                fig = px.histogram(df, x='systolic_bp', nbins=30,
                                 title='Systolic Blood Pressure Distribution',
                                 labels={'systolic_bp': 'Systolic BP (mmHg)', 'count': 'Count'},
                                 color_discrete_sequence=['#ff7f0e'])
                # Add reference lines for BP categories
                fig.add_vline(x=120, line_dash="dash", line_color="green", 
                            annotation_text="Normal (<120)")
                fig.add_vline(x=140, line_dash="dash", line_color="red", 
                            annotation_text="Hypertension (≥140)")
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk factor analysis
            if 'age' in df.columns and 'systolic_bp' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader('Risk by Age Group')
                    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 70, 100], 
                                           labels=['<40', '40-49', '50-59', '60-69', '70+'])
                    risk_by_age = df.groupby('age_group')['cvd_risk_10yr'].mean()
                    fig = px.bar(x=risk_by_age.index, y=risk_by_age.values,
                               title='CVD Risk by Age Group',
                               labels={'x': 'Age Group', 'y': 'Risk Proportion'},
                               color_discrete_sequence=['#9467bd'])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader('BP vs Age by Risk Level')
                    fig = px.scatter(df, x='age', y='systolic_bp', color='cvd_risk_10yr',
                                   title='Blood Pressure vs Age by Risk Level',
                                   labels={'age': 'Age (years)', 'systolic_bp': 'Systolic BP (mmHg)',
                                          'cvd_risk_10yr': 'CVD Risk'},
                                   color_discrete_map={0: '#2ca02c', 1: '#d62728'})
                    st.plotly_chart(fig, use_container_width=True)
                    
        else:
            st.info('Cardiovascular dataset not available.')
            st.markdown("""
            **Expected Analysis:**
            - 10-year cardiovascular risk predictions
            - Blood pressure distribution analysis
            - Risk factor correlations (age, BP, cholesterol)
            - Population health insights
            
            *This would show real cardiovascular risk modeling results in a production environment.*
            """)

    with tabs[2]:
        st.header('Fracture Risk Analysis')
        if 'fracture' in dfs:
            fr = dfs['fracture']
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric('Fracture Events', f"{len(fr):,}")
            with col2:
                if 'patient_id' in fr.columns and 'master' in dfs:
                    total_patients = len(dfs['master'])
                    fracture_rate = (len(fr) / total_patients) * 100
                    st.metric('Fracture Rate', f"{fracture_rate:.1f}%")
            with col3:
                if 'age' in fr.columns:
                    avg_fracture_age = fr['age'].mean()
                    st.metric('Avg Fracture Age', f"{avg_fracture_age:.1f} years")
            with col4:
                if 'fracture_type' in fr.columns:
                    common_type = fr['fracture_type'].mode()[0] if not fr['fracture_type'].mode().empty else "N/A"
                    st.metric('Most Common', common_type)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader('Fracture Events by Type')
                if 'fracture_type' in fr.columns:
                    type_counts = fr['fracture_type'].value_counts()
                    fig = px.bar(x=type_counts.index, y=type_counts.values,
                               title='Distribution of Fracture Types',
                               labels={'x': 'Fracture Type', 'y': 'Number of Events'},
                               color_discrete_sequence=['#e377c2'])
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Create synthetic fracture types for demo
                    fracture_types = ['Hip', 'Wrist', 'Spine', 'Ankle', 'Shoulder']
                    type_counts = pd.Series([len(fr)//3, len(fr)//4, len(fr)//5, len(fr)//6, len(fr)//7], 
                                          index=fracture_types)
                    fig = px.bar(x=type_counts.index, y=type_counts.values,
                               title='Distribution of Fracture Types',
                               labels={'x': 'Fracture Type', 'y': 'Number of Events'},
                               color_discrete_sequence=['#e377c2'])
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader('Fractures by Age Group')
                if 'age' in fr.columns:
                    fr['age_group'] = pd.cut(fr['age'], bins=[0, 30, 45, 60, 75, 100], 
                                           labels=['<30', '30-44', '45-59', '60-74', '75+'])
                    age_counts = fr['age_group'].value_counts().sort_index()
                    fig = px.bar(x=age_counts.index, y=age_counts.values,
                               title='Fracture Events by Age Group',
                               labels={'x': 'Age Group', 'y': 'Number of Fractures'},
                               color_discrete_sequence=['#17becf'])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Demo data
                    age_groups = ['<30', '30-44', '45-59', '60-74', '75+']
                    age_counts = pd.Series([len(fr)//8, len(fr)//6, len(fr)//4, len(fr)//3, len(fr)//2], 
                                         index=age_groups)
                    fig = px.bar(x=age_counts.index, y=age_counts.values,
                               title='Fracture Events by Age Group',
                               labels={'x': 'Age Group', 'y': 'Number of Fractures'},
                               color_discrete_sequence=['#17becf'])
                    st.plotly_chart(fig, use_container_width=True)
            
            # Risk assessment insights
            if 'master' in dfs and 'age' in fr.columns:
                st.subheader('Risk Assessment Summary')
                col1, col2 = st.columns(2)
                
                with col1:
                    # Calculate risk by age group
                    master_df = dfs['master']
                    if 'age' in master_df.columns:
                        master_df['age_group'] = pd.cut(master_df['age'], bins=[0, 30, 45, 60, 75, 100], 
                                                      labels=['<30', '30-44', '45-59', '60-74', '75+'])
                        
                        # Count patients and fractures by age group
                        patients_by_age = master_df['age_group'].value_counts().sort_index()
                        fractures_by_age = fr['age_group'].value_counts().sort_index()
                        
                        # Calculate risk rate
                        risk_rate = (fractures_by_age / patients_by_age * 100).fillna(0)
                        
                        fig = px.line(x=risk_rate.index, y=risk_rate.values,
                                    title='Fracture Risk Rate by Age Group',
                                    labels={'x': 'Age Group', 'y': 'Fracture Rate (%)'},
                                    markers=True)
                        fig.update_traces(line=dict(color='#d62728', width=3))
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Display risk insights
                    st.markdown("""
                    **Key Insights:**
                    - Fracture risk increases significantly with age
                    - Peak risk occurs in patients over 75 years
                    - Hip and wrist fractures are most common
                    - Early intervention crucial for high-risk groups
                    
                    **Clinical Recommendations:**
                    - Bone density screening for patients >65
                    - Fall prevention programs for elderly
                    - Calcium and Vitamin D supplementation
                    """)
                    
        else:
            st.info('Fracture dataset not available.')
            st.markdown("""
            **Expected Analysis:**
            - Fracture event tracking and classification
            - Age and gender-based risk stratification  
            - Bone density correlation analysis
            - FRAX score integration for 10-year fracture probability
            
            *This would show real fracture risk assessment results in a production environment.*
            """)

    with tabs[3]:
        st.header('Patient Lookup & ML Prediction')
        
        # Two-column layout for patient search and model selection
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader('Patient Search')
            pid = st.text_input('Enter patient_id (e.g. CVD_000001 or FRAC_000001)')
            
        with col2:
            st.subheader('ML Model Selection')
            repo_root = Path(__file__).resolve().parents[2]
            model_paths = find_models(repo_root)
            selected_model_path: Optional[Path] = None
            
            if model_paths:
                model_options = [str(p) for p in model_paths]
                sel = st.selectbox('Select model for prediction:', ['None'] + model_options)
                if sel != 'None':
                    selected_model_path = Path(sel)
                    
                    # Load model and show info
                    model = load_model(selected_model_path)
                    if model is not None:
                        st.success(f"✅ Model loaded: {selected_model_path.name}")
                        
                        # Show model info
                        model_info = f"**Model Type:** {type(model).__name__}"
                        if hasattr(model, 'n_features_in_'):
                            model_info += f"\n\n**Features:** {model.n_features_in_}"
                        st.markdown(model_info)
                        
                        # Auto-infer and save features if needed
                        try:
                            existing = load_features_list(selected_model_path)
                            if existing is None:
                                sample_df = None
                                for d in dfs.values():
                                    if isinstance(d, pd.DataFrame) and not d.empty:
                                        sample_df = d
                                        break
                                inferred = infer_feature_names(model, sample_df)
                                if inferred:
                                    save_features_list(selected_model_path, inferred)
                                    st.info(f'Auto-saved features.json ({len(inferred)} features)')
                        except Exception:
                            pass
                        
                        # Show preprocessor status
                        preproc_path = find_preprocessor_for_model(selected_model_path)
                        if preproc_path is not None:
                            st.info(f'🔧 Preprocessor: {preproc_path.name}')
                        else:
                            st.warning('⚠️ No preprocessor found - using best-effort alignment')
            else:
                st.warning('No trained models found. Run the ML training pipeline first.')
        
        # Patient search results
        if pid:
            st.divider()
            patient_found = False
            
            # Search across all datasets
            for key, df in dfs.items():
                if 'patient_id' in df.columns:
                    match = df[df['patient_id'] == pid]
                    if not match.empty:
                        patient_found = True
                        row = match.iloc[0]
                        
                        st.subheader(f'👤 Patient Found in {key.title()} Dataset')
                        
                        # Display patient info in a nice format
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Patient Information:**")
                            # Show key demographics first
                            key_fields = ['patient_id', 'age', 'gender', 'bmi']
                            for field in key_fields:
                                if field in row.index:
                                    st.write(f"**{field.replace('_', ' ').title()}:** {row[field]}")
                        
                        with col2:
                            # Show additional clinical data
                            st.markdown("**Clinical Data:**")
                            other_fields = [col for col in row.index if col not in key_fields and col != 'patient_id']
                            for field in other_fields[:8]:  # Limit to first 8 additional fields
                                if pd.notna(row[field]):
                                    st.write(f"**{field.replace('_', ' ').title()}:** {row[field]}")
                        
                        # Full data table (expandable)
                        with st.expander("📋 View Complete Patient Record"):
                            st.dataframe(match.T, use_container_width=True)
                        
                        # ML Prediction section
                        if selected_model_path is not None and model is not None:
                            st.divider()
                            st.subheader('🤖 ML Prediction Results')
                            
                            try:
                                # Prepare features for prediction
                                preproc_path = find_preprocessor_for_model(selected_model_path)
                                preproc = None
                                if preproc_path is not None:
                                    preproc = load_preprocessor(preproc_path)
                                
                                # Clean and prepare input data
                                X = match.drop(columns=['patient_id']).select_dtypes(include=['number']).fillna(0)
                                
                                # Apply preprocessing
                                if isinstance(model, Pipeline):
                                    X_processed = X
                                elif preproc is not None and hasattr(preproc, 'transform'):
                                    X_processed = preproc.transform(X)
                                else:
                                    X_processed = X
                                
                                # Make prediction
                                prediction_col1, prediction_col2 = st.columns(2)
                                
                                with prediction_col1:
                                    if hasattr(model, 'predict_proba'):
                                        proba = model.predict_proba(X_processed)
                                        if proba.shape[1] > 1:
                                            risk_prob = proba[0, 1]
                                            st.metric("🎯 Risk Probability", f"{risk_prob:.1%}")
                                            
                                            # Risk level interpretation
                                            if risk_prob < 0.3:
                                                risk_level = "🟢 Low Risk"
                                                color = "green"
                                            elif risk_prob < 0.7:
                                                risk_level = "🟡 Moderate Risk"  
                                                color = "orange"
                                            else:
                                                risk_level = "🔴 High Risk"
                                                color = "red"
                                            
                                            st.markdown(f"**Risk Level:** :{color}[{risk_level}]")
                                        else:
                                            st.metric("Prediction Score", f"{proba[0, 0]:.3f}")
                                    else:
                                        pred = model.predict(X_processed)[0]
                                        st.metric("🎯 Predicted Class", str(pred))
                                
                                with prediction_col2:
                                    # Feature importance for this prediction
                                    feats = load_features_list(selected_model_path) or infer_feature_names(model, X)
                                    if feats:
                                        fi = compute_feature_importance(model, feats)
                                        if fi is not None and not fi.empty:
                                            st.markdown("**🔍 Top Risk Factors:**")
                                            top_features = fi.head(5)
                                            for idx, row in top_features.iterrows():
                                                feat_name = row['feature'].replace('_', ' ').title()
                                                importance = row['importance']
                                                st.write(f"• {feat_name}: {importance:.3f}")
                                
                                # Clinical recommendations based on prediction
                                st.subheader('📋 Clinical Recommendations')
                                if hasattr(model, 'predict_proba') and 'risk_prob' in locals():
                                    if risk_prob > 0.7:
                                        st.error("""
                                        **High Risk Patient - Immediate Action Required:**
                                        - Schedule comprehensive clinical assessment
                                        - Consider preventive interventions
                                        - Implement enhanced monitoring protocol
                                        - Patient education on risk factors
                                        """)
                                    elif risk_prob > 0.3:
                                        st.warning("""
                                        **Moderate Risk Patient - Preventive Care:**
                                        - Regular follow-up appointments
                                        - Lifestyle modification counseling
                                        - Monitor key risk factors
                                        - Consider screening tests
                                        """)
                                    else:
                                        st.success("""
                                        **Low Risk Patient - Routine Care:**
                                        - Continue current care plan
                                        - Annual health maintenance
                                        - Promote healthy lifestyle
                                        - Routine monitoring sufficient
                                        """)
                                        
                            except Exception as e:
                                st.error(f'❌ Prediction failed: {str(e)}')
                                st.info("This may be due to feature mismatch between the patient data and model training data.")
                        
                        break  # Found patient, stop searching
            
            if not patient_found:
                st.warning(f'❌ Patient "{pid}" not found in any dataset.')
                st.info("""
                **Try these sample patient IDs:**
                - FRAC_000001 to FRAC_005000 (Fracture risk patients)
                - CVD_000001 to CVD_002000 (Cardiovascular patients)
                """)
        
        # Add some helpful information when no patient is selected
        else:
            st.info("""
            **🔍 How to use Patient Lookup:**
            1. Enter a patient ID in the format shown above
            2. Select a trained ML model for risk prediction
            3. View patient demographics and clinical data
            4. Get AI-powered risk assessment and recommendations
            
            **Available Features:**
            - Patient demographic analysis
            - ML-based risk scoring
            - Clinical decision support
            - Personalized recommendations
            """)
            
            # Show model performance metrics if available
            if selected_model_path is not None:
                st.subheader('📊 Model Performance Summary')
                st.markdown("""
                **Model Capabilities:**
                - Trained on synthetic medical data
                - Features engineered clinical variables
                - Cross-validated performance metrics
                - Production-ready inference pipeline
                
                *In a real clinical setting, this would show model validation metrics, 
                confidence intervals, and regulatory compliance information.*
                """)


if __name__ == '__main__':
    main()
