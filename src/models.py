from typing import List, Dict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.compose import ColumnTransformer


def build_models(
    task_type: str,
    preprocessor: ColumnTransformer,
    model_names: List[str],
    n_estimators: int = 100,
    random_state: int = 42,
) -> Dict[str, Pipeline]:

    models = {}

    if task_type == "regression":
        for m in model_names:
            if m == "linear":
                pipe = Pipeline([
                    ("preprocess", preprocessor),
                    ("model", LinearRegression()),
                ])
                models["LinearRegression"] = pipe

            if m == "rf":
                pipe = Pipeline([
                    ("preprocess", preprocessor),
                    ("model", RandomForestRegressor(
                        n_estimators=n_estimators,
                        random_state=random_state,
                    )),
                ])
                models["RandomForestRegressor"] = pipe

    elif task_type == "classification":
        for m in model_names:
            if m == "log_reg":
                pipe = Pipeline([
                    ("preprocess", preprocessor),
                    ("model", LogisticRegression(max_iter=1000)),
                ])
                models["LogisticRegression"] = pipe

            if m == "rf":
                pipe = Pipeline([
                    ("preprocess", preprocessor),
                    ("model", RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=random_state,
                    )),
                ])
                models["RandomForestClassifier"] = pipe

    return models
