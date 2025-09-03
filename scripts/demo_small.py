"""Small demo script — trains a tiny logistic regression on a small CSV (if present) or generates data.
Creates a saved model at .models/demo_small_model.joblib and prints AUC.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic" / "demo_small.csv"
MODEL_DIR = ROOT / ".models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded demo CSV: {DATA_PATH} — {len(df)} rows")
else:
    # fallback: generate tiny synthetic dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, n_redundant=0, random_state=42)
    cols = ["f1", "f2", "f3", "f4", "f5"]
    df = pd.DataFrame(X, columns=cols)
    df["outcome"] = y
    print("No demo CSV found; generated synthetic dataset (200 rows)")

# Normalize column names and prepare X, y
if "outcome" not in df.columns:
    raise SystemExit("demo data must contain 'outcome' column")

X = df.drop(columns=["outcome"]).select_dtypes(include=[np.number])
y = df["outcome"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f"Demo model AUC: {auc:.3f}")

out_path = MODEL_DIR / "demo_small_model.joblib"
joblib.dump(model, out_path)
print(f"Saved demo model to {out_path}")
