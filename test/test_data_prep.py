import pandas as pd
from src.data_prep import basic_clean, split_feature_types

def test_basic_clean_removes_nans_and_caps_outliers():
    df = pd.DataFrame({
        "age": [20, 30, None, 200],
        "income": [1000, None, 2000, 999999],
        "gender": ["F", "M", None, "F"],
        "target": [0, 1, 0, 1],
    })

    df_clean = basic_clean(
        df,
        target_col="target",
        numeric_strategy="mean",
        categorical_strategy="most_frequent",
        cap_outliers=True,
    )

    numeric_cols, categorical_cols = split_feature_types(df_clean, "target")

    assert df_clean[numeric_cols].isna().sum().sum() == 0
    assert df_clean[categorical_cols].isna().sum().sum() == 0
    assert df_clean["income"].max() < 999999
