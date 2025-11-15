import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def get_global_feature_importance(pipeline: Pipeline):
    """
    Extract feature importances from the model inside a pipeline.
    Supports:
    - feature_importances_ (tree models)
    - coef_ (linear / logistic)
    """

    model = pipeline.named_steps["model"]
    preproc = pipeline.named_steps["preprocess"]

    if hasattr(preproc, "get_feature_names_out"):
        names = preproc.get_feature_names_out()
    else:
        names = [f"feature_{i}" for i in range(model.n_features_in_)]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)

    else:
        raise ValueError("Model has no importances")

    return pd.DataFrame({
        "feature": names,
        "importance": importances
    }).sort_values("importance", ascending=False)
