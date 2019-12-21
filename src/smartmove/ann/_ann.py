from typing import Union
import logging

import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

from smartmove.ann.models import build_ff_regressor

logging.basicConfig()
_logger = logging.getLogger(__name__)


def log_cv_results(cv_obj, prefix: str = None):
    """
    Log parameters and metrics from CV model configurations
    """
    results = cv_obj.cv_results_

    for i in range(len(results["params"])):
        with mlflow.start_run(run_name=f"{prefix}_{i}" if prefix else f"cv_{i}"):
            for k, v in results.items():
                if k.startswith("param_"):
                    mlflow.log_param(k.replace("param_", ""), v.data[i])
                if k.startswith("split"):
                    mlflow.log_metric(k[7:], v[0], step=int(k[5]))
                if any([k.startswith(s) for s in ["mean_", "std_", "rank_"]]):
                    mlflow.log_metric(k, v[i])

        mlflow.end_run()


def build_model(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.DataFrame],
    build_fn: object,
    param_grid: dict,
    cv_kwargs: dict,
    epochs: int,
    batch_size=int,
):
    """
    Build the model, archiving it and results

    Parameters
    ----------
    estimator: estimator object
        Estimator with scikit-learn interface, document in `:func:sklearn.model_selection.GridSearchCV`.

    """
    sk_estimator = KerasRegressor(build_fn)

    # TODO use Random or Lassor or something
    gs = GridSearchCV(
        estimator=sk_estimator,
        param_grid=param_grid,
        return_train_score=True,
        **cv_kwargs,
    )
    gs.fit(X=X, y=y)

    log_cv_results(gs, prefix="grid-search")

    return gs


def build_tpot_model():
    """
    Possibly auto generate a pipeline to see what we can get ;)
    https://github.com/EpistasisLab/tpot
    """
    raise NotImplementedError()


if __name__ == "__main__":
    import mlflow
    import os
    from sklearn.metrics import r2_score, make_scorer, mean_squared_error

    from smartmove.ann.model_utils import random_data

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", None))

    X, y = random_data(n_samples=100, n_features=2)
    param_grid_ff = {
        "n_input_nodes": [X.shape[1]],
        "n_hidden_nodes": [10],
        "n_hidden_layers": [1, 3, 5],
        "n_output_nodes": [1],
        "l1": [0.1],
        "l2": [0.2],
        "dropout_rate": [0.2],
        "activation": ["tanh"],
        "loss": ["mse"],
        "optimizer": ["adam"],
    }

    # "metrics": ["accuracy", "mse"]
    cv_kwargs = {
        "scoring": {
            "r2": make_scorer(r2_score),
            "mse": make_scorer(mean_squared_error),
        },
        "refit": "r2",
    }

    epochs = 2
    batch_size = 10
    gs = build_model(
        X,
        y,
        build_ff_regressor,
        param_grid_ff,
        cv_kwargs,
        epochs=epochs,
        batch_size=batch_size,
    )
