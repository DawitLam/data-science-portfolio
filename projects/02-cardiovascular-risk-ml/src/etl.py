import pandas as pd
import numpy as np


def read_synthetic(path: str) -> pd.DataFrame:
    """Read a CSV synthetic dataset into a DataFrame."""
    return pd.read_csv(path)


def basic_etl(df: pd.DataFrame) -> pd.DataFrame:
    """Run lightweight ETL steps:
    - parse common date columns
    - compute age if possible
    - normalize simple categorical values
    - ensure numeric columns are numeric
    Returns a new DataFrame.
    """
    df = df.copy()

    # Parse common date columns if present
    for date_col in ("dob", "date_of_birth", "visit_date", "admission_date", "encounter_date"):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Compute age when dob and visit_date are available
    if "dob" in df.columns and "visit_date" in df.columns:
        df["age"] = ((df["visit_date"] - df["dob"]).dt.days / 365.25).astype(float)
    elif "dob" in df.columns and "visit_date" not in df.columns:
        df["age"] = ((pd.Timestamp.now() - df["dob"]).dt.days / 365.25).astype(float)

    # Normalize common categorical fields
    if "sex" in df.columns:
        df["sex"] = (
            df["sex"].astype(str).str.strip().str.upper().replace({"MALE": "M", "FEMALE": "F"})
        )

    # Ensure numeric columns are numeric
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Also try to coerce object columns that look numeric
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            coerced = pd.to_numeric(df[col], errors="coerce")
            # if a reasonable proportion converts, replace
            if coerced.notna().sum() > 0 and coerced.notna().sum() / max(1, len(coerced)) > 0.5:
                df[col] = coerced
        except Exception:
            pass

    # Coerce remaining numeric-looking columns
    for c in df.columns:
        if df[c].dtype == object:
            # leave as-is
            continue

    return df
