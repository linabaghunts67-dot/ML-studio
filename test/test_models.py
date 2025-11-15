import pandas as pd
from src.features import build_preprocessor
from src.models import build_models

def test_models_can_fit_and_predict_regression():
    df = pd.DataFrame({
        "x1": [1, 2, 3, 4, 5],
        "x2": [10, 20, 30, 40, 50],
        "target": [2.0, 4.1, 6.2, 8.1, 10.2],
    })

    preprocessor, _, _ = build_preprocessor(df, "target")
    models = build_models("regression", preprocessor, ["linear", "rf"], n_estimators=10)

    for name, model in models.items():
        model.fit(df[["x1", "x2"]], df["target"])
        preds = model.predict(df[["x1", "x2"]])
        assert len(preds) == len(df)

def test_models_can_fit_and_predict_classification():
    df = pd.DataFrame({
        "x1": [1, 2, 3, 4, 5, 6],
        "x2": [10, 20, 30, 40, 50, 60],
        "target": [0, 1, 0, 1, 0, 1],
    })

    preprocessor, _, _ = build_preprocessor(df, "target")
    models = build_models("classification", preprocessor, ["log_reg", "rf"], n_estimators=10)

    for name, model in models.items():
        model.fit(df[["x1", "x2"]], df["target"])
        preds = model.predict(df[["x1", "x2"]])
        assert len(preds) == len(df)
