import numpy as np
import pandas as pd
from typing import List, Tuple


def split_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def basic_clean(
    df: pd.DataFrame,
    target_col: str,
    numeric_strategy: str = "mean",
    categorical_strategy: str = "most_frequent",
    cap_outliers: bool = False,
) -> pd.DataFrame:
    df = df.copy()
    numeric_cols, categorical_cols = split_feature_types(df, target_col)

    if numeric_strategy == "drop rows with NaNs":
        df = df.dropna(subset=numeric_cols)
    else:
        if numeric_cols:
            if numeric_strategy == "mean":
                fill_vals = df[numeric_cols].mean()
            elif numeric_strategy == "median":
                fill_vals = df[numeric_cols].median()
            else:
                raise ValueError("Unknown numeric strategy")
            df[numeric_cols] = df[numeric_cols].fillna(fill_vals)

    if categorical_cols:
        if categorical_strategy == "most_frequent":
            fill_vals = df[categorical_cols].mode().iloc[0]
        elif categorical_strategy == "create 'Unknown' category":
            fill_vals = {col: "Unknown" for col in categorical_cols}
        else:
            raise ValueError("Unknown categorical strategy")
        df[categorical_cols] = df[categorical_cols].fillna(fill_vals)

    if cap_outliers and numeric_cols:
        for col in numeric_cols:
            lower = np.percentile(df[col], 1)
            upper = np.percentile(df[col], 99)
            df[col] = df[col].clip(lower, upper)

    return df
