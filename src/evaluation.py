import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder

from .features import build_preprocessor
from .models import build_models


def train_and_evaluate_models(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    model_names: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:

    df = df.dropna(subset=[target_col]).copy()
    X = df.drop(columns=[target_col])
    y = df[target_col]

    le = None
    if task_type == "classification" and (y.dtype == "O" or not np.issubdtype(y.dtype, np.number)):
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if task_type == "classification" else None
    )

    preprocessor, _, _ = build_preprocessor(df, target_col)
    models = build_models(task_type, preprocessor, model_names, n_estimators)

    metrics_rows = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        y_pred = model.predict(X_test)

        if task_type == "regression":
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            metrics_rows.append({
                "model": name,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
            })

        else:
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                roc = roc_auc_score(y_test, y_proba)
            except:
                roc = np.nan

            metrics_rows.append({
                "model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "ROC_AUC": roc,
            })

    metrics_df = pd.DataFrame(metrics_rows)

    test_df = X_test.copy()
    y_true = y_test

    if task_type == "classification" and le is not None:
        y_true = le.inverse_transform(y_true)

    test_df[target_col + "_true"] = y_true

    last_model_name = list(fitted_models.keys())[-1]
    last_model = fitted_models[last_model_name]
    y_pred_last = last_model.predict(X_test)

    if task_type == "classification" and le is not None:
        y_pred_last = le.inverse_transform(y_pred_last)

    test_df[target_col + "_pred"] = y_pred_last

    return metrics_df, fitted_models, test_df.reset_index(drop=True)
