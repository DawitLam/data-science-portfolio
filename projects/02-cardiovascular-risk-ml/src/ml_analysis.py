import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


def train_models(
    df: pd.DataFrame, target: str, features: list | None = None, test_size: float = 0.2, random_state: int = 42
) -> dict:
    """Train a small suite of models and return trained estimators and simple metrics.

    Returns a dict: { 'models': {name: estimator}, 'scores': {name: {'auc': ...}} }
    """
    df = df.copy()
    if features is None:
        features = df.select_dtypes(include=["number"]).columns.drop(target).tolist()

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y if len(np.unique(y))>1 else None, random_state=random_state)

    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=100, random_state=random_state),
    }

    try:
        import xgboost as xgb

        models["xgb"] = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)
    except Exception:
        # xgboost optional; silently skip if not installed
        pass

    trained = {}
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        # get probabilities when available
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_test)[:, 1]
        else:
            preds = model.predict(X_test)

        try:
            auc = float(roc_auc_score(y_test, preds))
        except Exception:
            auc = float("nan")

        scores[name] = {"auc": auc}

    return {"models": trained, "scores": scores, "X_test": X_test, "y_test": y_test}


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return a minimal evaluation dict for a fitted model."""
    if hasattr(model, "predict_proba"):
        preds = model.predict_proba(X_test)[:, 1]
    else:
        preds = model.predict(X_test)

    try:
        auc = float(roc_auc_score(y_test, preds))
    except Exception:
        auc = float("nan")

    acc = float(accuracy_score(y_test, (preds > 0.5).astype(int)))
    return {"auc": auc, "accuracy": acc}


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Return a DataFrame of feature importances or coefficients sorted by absolute importance.

    Supports scikit-learn tree ensembles (feature_importances_), xgboost, lightgbm, and linear models
    with `coef_`. Returns columns: feature, importance, sign (for coef-based models).
    """
    import pandas as pd
    import numpy as np

    feats = list(feature_names)

    # tree-based
    if hasattr(model, "feature_importances_"):
        imp = np.array(model.feature_importances_)
        df = pd.DataFrame({"feature": feats, "importance": imp})
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        df["sign"] = np.sign(df["importance"])
        return df

    # xgboost
    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            fmap = booster.get_score(importance_type="weight")
            # map to full list
            imp = [fmap.get(f, 0.0) for f in feats]
            df = pd.DataFrame({"feature": feats, "importance": imp}).sort_values("importance", ascending=False).reset_index(drop=True)
            df["sign"] = np.sign(df["importance"])
            return df
        except Exception:
            pass

    # linear / logistic
    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        # if multiclass, reduce by taking mean absolute or first class
        if coef.shape[0] != len(feats):
            # try to take mean absolute across classes
            coef = np.mean(model.coef_, axis=0)
        df = pd.DataFrame({"feature": feats, "importance": np.abs(coef), "coef": coef})
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        df["sign"] = df["coef"].apply(lambda x: "+" if x >= 0 else "-")
        return df

    # fallback: try permutation importance if sklearn available
    try:
        from sklearn.inspection import permutation_importance
        import numpy as np
        # small sample for speed
        X = np.zeros((1, len(feats)))
        result = permutation_importance(model, X, np.array([0]), n_repeats=5, random_state=0)
        imp = result.importances_mean
        df = pd.DataFrame({"feature": feats, "importance": imp}).sort_values("importance", ascending=False).reset_index(drop=True)
        df["sign"] = 0
        return df
    except Exception:
        # last resort: return features with zeros
        import pandas as pd
        df = pd.DataFrame({"feature": feats, "importance": [0.0] * len(feats)})
        df["sign"] = 0
        return df


def plot_interactive_roc(model, X_test: pd.DataFrame, y_test: pd.Series, save_html: str | None = None):
    """Create an interactive ROC curve using plotly and return the figure object.

    If plotly is not installed, raises ImportError with a helpful message. Optionally saves
    the interactive plot to `save_html`.
    """
    try:
        import plotly.graph_objects as go
        from sklearn.metrics import roc_curve, auc
    except Exception as e:
        raise ImportError("plot_interactive_roc requires plotly. Install with `pip install plotly`.") from e

    # get scores or predictions
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        # fall back to decision_function or predictions
        if hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            y_score = model.predict(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines+markers", name=f"ROC (AUC={roc_auc:.3f})", hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Chance"))
    fig.update_layout(title_text="Interactive ROC curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", width=800, height=600)

    if save_html:
        fig.write_html(save_html)

    return fig
