"""Small validator to load models and optional preprocessors and score a synthetic row.

Run this from the project folder using the virtualenv python to see which models and preprocessors load successfully.
"""
from pathlib import Path
import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

MODEL_GLOBS = [
    REPO_ROOT / 'models' / '*.pkl',
    REPO_ROOT / 'projects' / '01-fracture-risk-ml' / 'models' / 'trained_models' / '*.pkl',
    REPO_ROOT / 'projects' / '02-cardiovascular-risk-ml' / 'models' / '*.pkl',
]

CANDIDATES = []
for g in MODEL_GLOBS:
    CANDIDATES.extend(list(Path(g).parent.glob(Path(g).name)))

if not CANDIDATES:
    print('No model pkl files found in expected locations')

for p in CANDIDATES:
    print(f'--- Testing model: {p}')
    try:
        m = joblib.load(str(p))
        print('Loaded model object:', type(m))
    except Exception as e:
        print('Failed to load model:', e)
        continue

    # try find preprocessor
    pre_candidates = [p.parent / 'preprocessor.pkl', p.parent / 'scaler.pkl', p.parent / 'pipeline.pkl', p.parent / 'preprocessor.joblib']
    pre = None
    for pp in pre_candidates:
        if pp.exists():
            try:
                pre = joblib.load(str(pp))
                print('Loaded preprocessor:', pp.name, type(pre))
                break
            except Exception as e:
                print('Found preprocessor file but failed to load:', pp, e)

    # build a fake numeric row
    import numpy as np
    X = pd.DataFrame([np.zeros(5)], columns=[f'f{i}' for i in range(5)])
    try:
        if pre is not None and hasattr(pre, 'transform'):
            X2 = pre.transform(X)
        else:
            X2 = X
        # if pipeline model, pass raw X
        if hasattr(m, 'predict_proba'):
            print('Predicted proba shape:', m.predict_proba(X2).shape)
        else:
            print('Predicted label shape:', m.predict(X2).shape)
    except Exception as e:
        print('Scoring failed:', e)

print('Done')
