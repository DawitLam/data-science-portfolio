import pandas as pd
import numpy as np
from scipy import stats


def describe(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Provide descriptive statistics. If `by` is provided, return groupby describe."""
    if by and by in df.columns:
        return df.groupby(by).describe()
    return df.describe(include="all").transpose()


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Return correlation matrix for numeric columns."""
    return df.select_dtypes(include=["number"]).corr(method=method)


def t_test_groups(df: pd.DataFrame, group_col: str, target_col: str) -> dict:
    """Run an independent t-test between two groups for a numeric target.
    Returns a small dict with t-stat and p-value.
    """
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("t_test_groups currently supports exactly 2 groups")

    g1 = df[df[group_col] == groups[0]][target_col].dropna()
    g2 = df[df[group_col] == groups[1]][target_col].dropna()
    stat, p = stats.ttest_ind(g1, g2, equal_var=False)
    return {"group_names": tuple(groups), "t_stat": float(stat), "p_value": float(p)}
