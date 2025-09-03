"""Lightweight end-to-end pipeline for the cardiovascular risk project.

Usage (PowerShell):
  cd projects\02-cardiovascular-risk-ml
  & "C:/Users/Dama/Documents/Python project/portfolio/data-science-portfolio/.venv/Scripts/python.exe" run_pipeline.py --target target

The script will:
 - read a CSV (defaults to repository `demo_small.csv` if present)
 - run `basic_etl`, `impute_missing`, `remove_outliers_zscore`
 - train a small model suite (logistic, rf, optional xgboost)
 - save the best model and a `training_results.json` under `models/`
"""

from pathlib import Path
import argparse
import json
import joblib
from datetime import datetime
import importlib


def main(input_csv: Path, target: str, model_dir: Path, test_size: float = 0.2, random_state: int = 42):
    # Lazy imports from project modules
    try:
        from src import etl, cleaning, ml_analysis
    except Exception:
        # Fallback to package-style import when running from repo root
        import importlib
        etl = importlib.import_module('projects.02-cardiovascular-risk-ml.src.etl')
        cleaning = importlib.import_module('projects.02-cardiovascular-risk-ml.src.cleaning')
        ml_analysis = importlib.import_module('projects.02-cardiovascular-risk-ml.src.ml_analysis')

    print(f"Reading data from: {input_csv}")
    df = etl.read_synthetic(str(input_csv))

    print("Running basic ETL...")
    df = etl.basic_etl(df)

    print("Imputing missing values...")
    df = cleaning.impute_missing(df)

    print("Removing outliers (z-score)...")
    df = cleaning.remove_outliers_zscore(df)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data. Available columns: {df.columns.tolist()}")

    print("Training models...")
    results = ml_analysis.train_models(df, target=target, test_size=test_size, random_state=random_state)

    scores = results.get('scores', {})
    models = results.get('models', {})

    # select best by auc (nan-safe)
    best_name = None
    best_auc = float('-inf')
    for name, sc in scores.items():
        auc = sc.get('auc', float('nan'))
        try:
            if not (auc != auc):  # not nan
                if auc > best_auc:
                    best_auc = auc
                    best_name = name
        except Exception:
            continue

    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    artifact_summary = {
        'input_csv': str(input_csv),
        'target': target,
        'timestamp': timestamp,
        'scores': scores,
        'best_model': best_name,
        'best_auc': None if best_name is None else scores.get(best_name, {}).get('auc')
    }

    # save all models (lightweight)
    for name, model in models.items():
        path = model_dir / f"{name}_{timestamp}.pkl"
        joblib.dump(model, path)
        print(f"Saved model: {path}")

    # Save a copy of the best model as `best_model.pkl`
    if best_name:
        best_path = model_dir / 'best_model.pkl'
        joblib.dump(models[best_name], best_path)
        print(f"Saved best model ({best_name}) to {best_path}")

    # Save results json
    results_path = model_dir / f"training_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(artifact_summary, f, indent=2)

    print(f"Training complete. Artifacts saved to: {model_dir}")
    print(json.dumps(artifact_summary, indent=2))


if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parents[2]
    default_demo = repo_root / 'demo_small.csv'

    parser = argparse.ArgumentParser(description='Run lightweight cardiovascular pipeline')
    parser.add_argument('--input-csv', type=Path, default=(default_demo if default_demo.exists() else None), help='Path to input CSV')
    parser.add_argument('--use-shared-generator', action='store_true', help='If set, generate synthetic cardiovascular data using shared generator when no --input-csv is provided')
    parser.add_argument('--generate-n', type=int, default=2000, help='Number of synthetic rows to generate when using the shared generator')
    parser.add_argument('--generated-filename', type=str, default='cardiovascular_risk_data.csv', help='Filename to save generated synthetic data under data/synthetic/')
    parser.add_argument('--target', type=str, required=True, help='Name of the target column to predict')
    parser.add_argument('--model-dir', type=Path, default=Path('models'), help='Directory to save models and results')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)

    args = parser.parse_args()

    # If the user requested the shared generator and no explicit CSV was provided,
    # call the shared cardiovascular generator to create a small synthetic dataset
    # saved under data/synthetic/ and use it as the input for the pipeline.
    generated_temp_path = None
    if args.input_csv is None and args.use_shared_generator:
        synthetic_dir = repo_root / 'data' / 'synthetic'
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        output_path = synthetic_dir / args.generated_filename
        print(f"No input CSV provided. Generating synthetic data to: {output_path}")
        try:
            gen_mod = importlib.import_module('shared.data_generators.cardiovascular_data_generator')
            # generate and save
            gen_mod.generate_cardiovascular_data(n_samples=args.generate_n, output_path=str(output_path))
            generated_temp_path = output_path
        except Exception as e:
            raise RuntimeError('Failed to import or run shared generator: ' + str(e))

    # Final input selection: explicit CSV > generated dataset > demo_small.csv
    final_input = args.input_csv if args.input_csv is not None else (generated_temp_path if generated_temp_path is not None else None)

    if final_input is None:
        raise ValueError('No input CSV provided and no demo_small.csv found. Use --use-shared-generator or pass --input-csv')

    main(final_input, args.target, args.model_dir, test_size=args.test_size, random_state=args.random_state)
