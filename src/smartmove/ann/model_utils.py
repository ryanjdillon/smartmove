from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def random_data(n_samples=100, n_features=2):
    """Create random data"""
    data = {f"f{i}": np.random.random(n_samples) for i in range(n_features)}
    X = pd.DataFrame(data, index=range(n_samples)).values
    y = np.random.random(n_samples) * 100

    return X, y


def model_to_key(pipe: Pipeline, params: dict) -> str:
    import uuid

    return uuid.uuid4().hex


def key_to_model(key: str) -> Tuple[Pipeline, dict]:
    # Query Mlflow
    # return pipeline, params
    pass


def mlflow_results_exist(key: str):
    return False


def _save_model(model, params):
    # https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations
    # Maybe just configure model to be saved with MLflow, while documenting
    # those things mentioned in list above.
    raise NotImplementedError()
