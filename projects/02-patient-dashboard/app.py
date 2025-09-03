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
        st.header('Cohort overview')
        if 'master' in dfs:
            df = dfs['master']
            st.metric('Patients', f"{len(df):,}")
            st.subheader('Age distribution')
            st.bar_chart(df['age'].value_counts().sort_index())
        else:
            st.info('Master patient dataset not found. The app attempted to generate synthetic data if available.')

    with tabs[1]:
        st.header('Cardiovascular risk')
        if 'cvd' in dfs:
            df = dfs['cvd']
            st.metric('Records', f"{len(df):,}")
            st.subheader('10-year high-risk prevalence')
            st.write(df['cvd_risk_10yr'].value_counts(normalize=True).rename('proportion'))
            st.subheader('Systolic BP distribution')
            st.histogram = st.bar_chart(df['systolic_bp'].value_counts().sort_index())
        else:
            st.info('Cardiovascular dataset not available.')

    with tabs[2]:
        st.header('Fracture risk')
        if 'fracture' in dfs:
            fr = dfs['fracture']
            st.metric('Fracture events', f"{len(fr):,}")
            st.subheader('Fracture events by type')
            if 'fracture_type' in fr.columns:
                st.bar_chart(fr['fracture_type'].value_counts())
        else:
            st.info('Fracture dataset not available.')

    with tabs[3]:
        st.header('Patient lookup')
        pid = st.text_input('Enter patient_id (e.g. CVD_000001 or FRAC_000001)')
        # model selector
        repo_root = Path(__file__).resolve().parents[2]
        model_paths = find_models(repo_root)
        selected_model_path: Optional[Path] = None
        if model_paths:
            sel = st.selectbox('Load saved model for prediction (optional)', [str(p) for p in model_paths])
            if sel:
                selected_model_path = Path(sel)
                model = load_model(selected_model_path)
                # try to infer and persist feature ordering for this model if missing
                try:
                    if selected_model_path is not None and model is not None:
                        existing = load_features_list(selected_model_path)
                        if existing is None:
                            # attempt to infer from model or data
                            sample_df = None
                            # pick a sample dataframe from loaded dfs if present
                            for d in dfs.values():
                                if isinstance(d, pd.DataFrame) and not d.empty:
                                    sample_df = d
                                    break
                            inferred = infer_feature_names(model, sample_df)
                            if inferred:
                                save_features_list(selected_model_path, inferred)
                                st.caption(f'Inferred and saved features.json ({len(inferred)} features) next to model')
                except Exception:
                    pass
                # show preprocessor and feature info
                preproc_path = find_preprocessor_for_model(selected_model_path)
                if preproc_path is not None:
                    st.caption(f'Preprocessor found: {preproc_path.name} (loaded next to model)')
                else:
                    st.caption('No preprocessor file found next to model; attempting best-effort alignment')

                features_from_file = load_features_list(selected_model_path)
                if features_from_file:
                    st.caption(f'Loaded feature order from features.json ({len(features_from_file)} features)')
                else:
                    st.caption('No features.json found; attempting to infer feature ordering from model or data')
        else:
            st.info('No saved models (best_model.pkl) found in repository. Run the training pipeline to create models.')
        if pid:
            row = None
            # search across loaded tables
            for key, df in dfs.items():
                if 'patient_id' in df.columns:
                    match = df[df['patient_id'] == pid]
                    if not match.empty:
                        st.subheader(f'Found in {key} dataset')
                        st.write(match.T)
                        row = match.iloc[0]
                        # If model loaded, attempt prediction for matching row
                        if selected_model_path is not None and model is not None:
                            # try to locate and load a matching preprocessor
                            preproc_path = find_preprocessor_for_model(selected_model_path)
                            preproc = None
                            if preproc_path is not None:
                                preproc = load_preprocessor(preproc_path)

                            try:
                                # align features: drop non-numeric and id columns, fill nans
                                X = match.drop(columns=['patient_id']).select_dtypes(include=['number']).fillna(0)

                                # If the loaded model is a sklearn Pipeline, it may include preprocessing itself
                                if isinstance(model, Pipeline):
                                    # pipeline will handle preprocessing
                                    X_in = X
                                else:
                                    # if we found a separate preprocessor, use it; else try to use scaler in project 01 locations
                                    if preproc is not None:
                                        try:
                                            # if preproc is a transformer or pipeline, transform X
                                            if hasattr(preproc, 'transform'):
                                                X_in = preproc.transform(X)
                                            else:
                                                X_in = X
                                        except Exception:
                                            X_in = X
                                    else:
                                        X_in = X

                                # scoring
                                # compute and show feature importance if available
                                feats = load_features_list(selected_model_path) or infer_feature_names(model, X)
                                if feats:
                                    fi = compute_feature_importance(model, feats)
                                    if fi is not None and not fi.empty:
                                        st.subheader('Top features')
                                        st.table(fi.head(10))

                                if hasattr(model, 'predict_proba'):
                                    prob = model.predict_proba(X_in)[:, 1][0]
                                    st.success(f'Predicted probability (positive class): {prob:.3f}')
                                else:
                                    pred = model.predict(X_in)[0]
                                    st.success(f'Predicted label: {pred}')
                            except Exception as e:
                                st.warning(f'Could not score patient with model: {e}')
            if row is None:
                st.warning('Patient not found in current synthetic datasets.')


if __name__ == '__main__':
    main()
