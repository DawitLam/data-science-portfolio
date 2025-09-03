"""Scan saved models and try to infer and persist feature ordering as features.json next to each model.

Strategy, in order:
 - If model has `feature_names_in_`, use that.
 - If model is a sklearn Pipeline, check last estimator for `feature_names_in_`.
 - Otherwise, search nearby project `data` or repo `data/synthetic` for a CSV and use its numeric columns.
 - Only act on objects that appear to be estimators (have predict or predict_proba).
"""
from pathlib import Path
import joblib
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
print('Repo root:', REPO_ROOT)

# Candidate model locations
candidates = set()
# repo-level models folder
candidates.update(REPO_ROOT.glob('models/**/*.pkl'))
# project-level models
candidates.update(REPO_ROOT.glob('projects/**/models/**/*.pkl'))

candidates = sorted([p for p in candidates if p.is_file()])
print(f'Found {len(candidates)} candidate .pkl files')

from sklearn.pipeline import Pipeline
import pandas as pd

saved = []
skipped = []

for p in candidates:
    try:
        obj = joblib.load(str(p))
    except Exception as e:
        print(f'Could not load {p}: {e}')
        skipped.append((p, 'load-failed'))
        continue

    # skip pure transformers
    if not (hasattr(obj, 'predict') or hasattr(obj, 'predict_proba')):
        skipped.append((p, 'not-estimator'))
        continue

    features = None
    # 1: feature_names_in_
    try:
        if hasattr(obj, 'feature_names_in_'):
            features = list(getattr(obj, 'feature_names_in_'))
    except Exception:
        pass

    # 2: pipeline last estimator
    if features is None:
        try:
            if isinstance(obj, Pipeline):
                last = obj.steps[-1][1]
                if hasattr(last, 'feature_names_in_'):
                    features = list(getattr(last, 'feature_names_in_'))
        except Exception:
            pass

    # 3: search for nearby CSVs
    if features is None:
        # look up from model dir up to repo root for data folders
        for ancestor in [p.parent, *p.parents]:
            if ancestor == REPO_ROOT.parent:
                break
            # check ancestor/data and repo-level data/synthetic
            checks = [ancestor / 'data', ancestor / 'data' / 'synthetic']
            for c in checks:
                if c.exists() and c.is_dir():
                    # pick first csv
                    csvs = list(c.glob('**/*.csv'))
                    if csvs:
                        try:
                            df = pd.read_csv(csvs[0], nrows=10)
                            features = list(df.select_dtypes(include=['number']).columns)
                            if 'patient_id' in features:
                                # keep order but remove id
                                features = [f for f in features if f != 'patient_id']
                            break
                        except Exception:
                            continue
            if features:
                break

    if features is None or len(features) == 0:
        print(f'SKIP (no features inferred) for {p}')
        skipped.append((p, 'no-features'))
        continue

    out = p.parent / 'features.json'
    try:
        with open(out, 'w', encoding='utf8') as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
        print(f'SAVED features.json next to {p} ({len(features)} features)')
        saved.append((p, len(features)))
    except Exception as e:
        print(f'Failed to save features.json for {p}: {e}')
        skipped.append((p, 'save-failed'))

print('\nSummary:')
print('Saved:', len(saved))
print('Skipped:', len(skipped))
for s in skipped[:10]:
    print(' -', s[0].name, s[1])

if len(skipped) > 10:
    print('... (more skipped)')

print('Done')
