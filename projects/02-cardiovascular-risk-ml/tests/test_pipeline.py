import os
import sys
import numpy as np
import importlib.util
import importlib.machinery


# make repo root importable
TEST_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(TEST_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from shared.data_generators.cardiovascular_data_generator import generate_cardiovascular_data


def load_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    return module


PKG = os.path.join(REPO_ROOT, "projects", "02-cardiovascular-risk-ml", "src")
etl = load_module_from_path("etl", os.path.join(PKG, "etl.py"))
cleaning = load_module_from_path("cleaning", os.path.join(PKG, "cleaning.py"))
stats = load_module_from_path("stats", os.path.join(PKG, "stats.py"))
ml_analysis = load_module_from_path("ml_analysis", os.path.join(PKG, "ml_analysis.py"))


def test_full_pipeline_runs():
    # generate small synthetic dataset
    df = generate_cardiovascular_data(n_samples=200)

    # ETL
    df = etl.basic_etl(df)

    # cleaning
    df = cleaning.impute_missing(df)

    # quick stats
    desc = stats.describe(df)
    assert hasattr(desc, "transpose") or isinstance(desc, (dict,))

    # prefer the canonical CVD target if present
    if "cvd_risk_10yr" in df.columns:
        target = "cvd_risk_10yr"
    else:
        # find a binary target automatically (allow booleans)
        candidate_targets = [
            c
            for c in df.columns
            if df[c].dropna().nunique() == 2 and df[c].dtype in [np.int64, np.int32, int, bool, "bool"]
        ]
        assert candidate_targets, "No binary target found in synthetic data"
        target = candidate_targets[0]

    # If target is single-class (rare on synthetic draws), inject minor variation
    if df[target].nunique() == 1:
        # flip 10% of labels to the opposite class
        vals = df[target].unique()
        if len(vals) == 1:
            single = vals[0]
            other = 0 if single == 1 else 1
            n_flip = max(1, int(len(df) * 0.1))
            idx = df.sample(n=n_flip, random_state=42).index
            df.loc[idx, target] = other

    # Train tiny ML pipeline
    out = ml_analysis.train_models(df, target=target, test_size=0.2)
    assert "models" in out and "scores" in out
    # expect at least logistic and rf
    assert any(k in out["models"] for k in ["logistic", "rf"]) 

    # simple AUC sanity check
    for v in out["scores"].values():
        assert ("auc" in v) and (np.isnan(v["auc"]) or (0.0 <= v["auc"] <= 1.0))
