"""Demo: load a model, its features.json and preprocessor, score a real synthetic patient row and print results."""
from pathlib import Path
import joblib
import json
import pandas as pd
import re

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO_ROOT / 'models' / 'best_model.pkl'

def find_preprocessor_for_model(model_path: Path):
    candidates = [
        'preprocessor.pkl', 'preprocessor.joblib', 'scaler.pkl', 'scaler.joblib', 'pipeline.pkl', 'feature_pipeline.pkl'
    ]
    for d in (model_path.parent, model_path.parent.parent):
        for name in candidates:
            p = d / name
            if p.exists():
                return p
    return None


def load_features_list(model_path: Path):
    p = model_path.parent / 'features.json'
    if p.exists():
        try:
            return json.load(open(p, 'r', encoding='utf8'))
        except Exception:
            return None
    return None


def main():
    print('Repo root:', REPO_ROOT)
    if not MODEL_PATH.exists():
        print('Model not found at', MODEL_PATH)
        return

    print('Loading model:', MODEL_PATH)
    model = joblib.load(str(MODEL_PATH))

    features = load_features_list(MODEL_PATH)
    print('Loaded features.json:', bool(features))
    if features:
        print('First features:', features[:10])

    # find a sample patient row
    synth_dir = REPO_ROOT / 'data' / 'synthetic'
    df = None
    if synth_dir.exists():
        # prefer cardiovascular
        candidates = ['cardiovascular_risk_data.csv', 'master_patient_data.csv', 'fracture_events.csv']
        for name in candidates:
            p = synth_dir / name
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    print('Using dataset:', p.name, 'rows=', len(df))
                    break
                except Exception as e:
                    print('Failed loading', p, e)
    if df is None:
        # try repo-level data folder
        repo_data = REPO_ROOT / 'data'
        if repo_data.exists():
            csvs = list(repo_data.glob('**/*.csv'))
            if csvs:
                df = pd.read_csv(csvs[0])
                print('Using dataset:', csvs[0].name, 'rows=', len(df))

    if df is None or df.empty:
        print('No synthetic dataset found to demo with. Create data in data/synthetic first.')
        return

    # pick first patient row
    row = df.iloc[0]
    print('Demo patient index 0; patient_id:', row.get('patient_id', row.get('patient', '<no id>')))

    # build X (robust mapping + synthesis)
    def infer_feature_value(feature_name, row, df):
        # direct match
        if feature_name in df.columns:
            return row[feature_name]

        # common name variants
        lname = feature_name.lower()
        # id / patient id
        if lname in ('id', 'patient_id', 'patient'):
            return row.get('patient_id', row.get('patient', None))

        # blood pressure
        if 'systolic' in lname:
            # look for bp columns
            for cand in ['systolic_bp', 'sbp', 'blood_pressure_systolic', 'systolic']:
                if cand in df.columns:
                    return row[cand]
            # try split from blood_pressure column like "120/80"
            if 'blood_pressure' in df.columns:
                val = row['blood_pressure']
                if isinstance(val, str) and '/' in val:
                    try:
                        return int(val.split('/')[0])
                    except Exception:
                        pass
            # fallback reasonable default
            return 130

        if 'diastolic' in lname:
            for cand in ['diastolic_bp', 'dbp', 'blood_pressure_diastolic', 'diastolic']:
                if cand in df.columns:
                    return row[cand]
            if 'blood_pressure' in df.columns:
                val = row['blood_pressure']
                if isinstance(val, str) and '/' in val:
                    try:
                        return int(val.split('/')[1])
                    except Exception:
                        pass
            return 80

        # cholesterol
        if 'cholesterol' in lname:
            for cand in ['total_cholesterol', 'cholesterol_total', 'cholesterol']:
                if cand in df.columns:
                    return row[cand]
            return 200

        # diabetes flag: try to parse comorbidities or diagnosis columns
        if 'diabetes' in lname:
            for cand in ['diabetes', 'has_diabetes']:
                if cand in df.columns:
                    val = row[cand]
                    if pd.isna(val):
                        return 0
                    if isinstance(val, (int, float)):
                        return int(val)
                    if isinstance(val, str):
                        return 1 if val.lower() in ('1', 'true', 'yes') else 0
            # parse free-text comorbidities
            for cand in ['comorbidities', 'current_medications', 'conditions']:
                if cand in df.columns:
                    txt = str(row[cand]).lower()
                    return 1 if 'diabetes' in txt else 0
            return 0

        # age fallback
        if lname == 'age' and 'age' in df.columns:
            return row['age']

        # default: try to find any similarly named column
        for c in df.columns:
            if c.lower() == feature_name.lower():
                return row[c]

        # not found: return 0 for numeric features
        return 0

    if features:
        # construct a one-row dataframe with columns in 'features'
        built = {}
        missing = []
        def coerce_numeric(v):
            # try float
            try:
                return float(v)
            except Exception:
                pass
            # if string with digits, extract first group
            if isinstance(v, str):
                m = re.search(r"(\d+)", v)
                if m:
                    try:
                        return float(m.group(1))
                    except Exception:
                        return 0.0
            return 0.0

        for f in features:
            val = infer_feature_value(f, row, df)
            if pd.isna(val):
                val = 0
            # coerce strings like 'PT000001' to a numeric id or 0 for non-numeric
            try:
                num_val = coerce_numeric(val)
                built[f] = num_val
            except Exception:
                built[f] = val
            if f not in df.columns:
                missing.append(f)

        if missing:
            print('Warning: dataset missing features required by model; inferred/synthesized:', missing)

        # preserve feature names exactly as the model expects; avoid dropping non-numeric columns
        X = pd.DataFrame([built])[features].fillna(0)
    else:
        # fallback: numeric columns
        X = df.select_dtypes(include=['number']).iloc[[0]].fillna(0)

    # load preprocessor if present
    pre = None
    pre_path = find_preprocessor_for_model(MODEL_PATH)
    if pre_path:
        try:
            pre = joblib.load(str(pre_path))
            print('Loaded preprocessor:', pre_path.name)
        except Exception as e:
            print('Failed to load preprocessor', pre_path, e)

    # apply preprocessor if appropriate
    X_in = X
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            X_in = X
        else:
            if pre is not None and hasattr(pre, 'transform'):
                X_in = pre.transform(X)
    except Exception:
        pass

    # predict
    try:
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X_in)[:, 1][0]
            print(f'Predicted probability (positive class): {prob:.4f}')
        else:
            pred = model.predict(X_in)[0]
            print('Predicted label:', pred)
    except Exception as e:
        print('Scoring failed:', e)

if __name__ == '__main__':
    main()
