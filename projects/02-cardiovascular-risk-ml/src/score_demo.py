"""Load saved model artifact and score a small sample from synthetic generator.

Usage:
python projects/02-cardiovascular-risk-ml/src/score_demo.py
"""
import sys
from pathlib import Path
import joblib
import json

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = Path(REPO_ROOT) / "projects" / "02-cardiovascular-risk-ml" / "models"

# add repo root and src path for local imports
SRC = str(Path(__file__).resolve().parents[0])
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from shared.data_generators.cardiovascular_data_generator import generate_cardiovascular_data
import etl
import cleaning

print("Loading metadata...")
meta_path = MODELS_DIR / "metadata.json"
if not meta_path.exists():
    raise SystemExit("No metadata.json found — run save_and_demo.py first")

meta = json.loads(meta_path.read_text())
model_file = MODELS_DIR / f"cardio_best_{meta['best_model']}.joblib"
if not model_file.exists():
    raise SystemExit(f"Model file not found: {model_file}")

print(f"Loading model: {model_file}")
obj = joblib.load(model_file)
model = obj.get("model")
features = obj.get("features")

# generate a tiny sample
df = generate_cardiovascular_data(n_samples=5)
df = etl.basic_etl(df)
df = cleaning.impute_missing(df)
X = df[features]

print("Scoring sample records:")
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X)[:, 1]
    for i, p in enumerate(probs):
        print(f"sample {i}: P(high CVD risk) = {p:.3f}")
else:
    preds = model.predict(X)
    for i, p in enumerate(preds):
        print(f"sample {i}: pred = {p}")
