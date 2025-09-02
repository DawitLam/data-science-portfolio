import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def impute_missing(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
) -> pd.DataFrame:
    """Impute missing values for numeric and categorical columns.
    Uses sklearn SimpleImputer under the hood.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if num_cols:
        imp_num = SimpleImputer(strategy=numeric_strategy)
        df[num_cols] = imp_num.fit_transform(df[num_cols])

    if cat_cols:
        imp_cat = SimpleImputer(strategy=categorical_strategy, fill_value="missing")
        df[cat_cols] = imp_cat.fit_transform(df[cat_cols])

    return df


def remove_outliers_zscore(df: pd.DataFrame, cols: list | None = None, z_thresh: float = 3.0) -> pd.DataFrame:
    """Remove rows where numeric columns exceed z-score threshold.
    Returns a filtered DataFrame.
    """
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not cols:
        return df

    from scipy import stats

    mask = pd.Series(True, index=df.index)
    for c in cols:
        col = df[c].fillna(df[c].median())
        z = (col - col.mean()) / (col.std(ddof=0) if col.std(ddof=0) != 0 else 1.0)
        mask = mask & (z.abs() < z_thresh)

    return df.loc[mask]
