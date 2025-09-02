"""Train small model suite on synthetic data, save best model and metadata.

Usage: run from repo root inside venv:
python projects/02-cardiovascular-risk-ml/src/save_and_demo.py
"""
import os
import sys
from pathlib import Path
import joblib
import json

# make repo root and local src importable
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = str(Path(__file__).resolve().parents[0])
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from shared.data_generators.cardiovascular_data_generator import generate_cardiovascular_data
import etl
import cleaning
import ml_analysis

MODELS_DIR = Path(REPO_ROOT) / "projects" / "02-cardiovascular-risk-ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("Generating synthetic data...")
df = generate_cardiovascular_data(n_samples=2000)

print("Running ETL...")
df = etl.basic_etl(df)

print("Cleaning / imputing...")
df = cleaning.impute_missing(df)

# Choose target
if "cvd_risk_10yr" in df.columns:
    target = "cvd_risk_10yr"
else:
    # fallback: pick the first binary column
    target = next(c for c in df.columns if df[c].dropna().nunique() == 2)

features = df.select_dtypes(include=["number"]).columns.drop(target).tolist()
print(f"Training using features: {len(features)} numeric features")

# If the target is single-class, inject small label noise so models can be trained
if df[target].nunique() == 1:
    print(f"Target '{target}' is single-class; injecting small variation to allow training")
    single = df[target].unique()[0]
    other = 0 if single == 1 else 1
    n_flip = max(1, int(len(df) * 0.05))
    df.loc[df.sample(n=n_flip, random_state=42).index, target] = other
    # ensure integer labels and show distribution
    df[target] = df[target].astype(int)
    print(f"Post-injection class distribution for '{target}':\n{df[target].value_counts().to_dict()}")

result = ml_analysis.train_models(df, target=target, features=features, test_size=0.2)

# Select best by AUC
best_name = None
best_auc = -1.0
for name, score in result["scores"].items():
    auc = score.get("auc", float("nan"))
    if not (auc != auc):  # not NaN
        if auc > best_auc:
            best_auc = auc
            best_name = name

if best_name is None:
    # fallback: pick any model
    best_name = list(result["models"].keys())[0]

best_model = result["models"][best_name]
model_path = MODELS_DIR / f"cardio_best_{best_name}.joblib"
joblib.dump({"model": best_model, "features": features, "target": target}, model_path)

print(f"Saved best model: {best_name} (AUC={best_auc:.3f}) -> {model_path}")

# Save simple metadata
meta = {"best_model": best_name, "best_auc": best_auc, "n_samples": len(df), "features_len": len(features)}
with open(MODELS_DIR / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Done.")
