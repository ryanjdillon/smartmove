from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import L1L2


def build_linear_regressor():
    raise NotImplementedError()


def build_svn_regressor():
    raise NotImplementedError()


def build_randomforest_regressor():
    raise NotImplementedError()


def build_ff_regressor(
    n_input_nodes: int,
    n_hidden_nodes: int,
    n_hidden_layers: int,
    n_output_nodes: int = 1,
    l1: float = 0.1,
    l2: float = 0.2,
    dropout_rate: float = 0.2,
    activation: str = "tanh",
    loss: str = "mse",
    optimizer: str = "adam",
):
    """
    Feed forward regressor model constructor

    Also refered to in the sklearn API documentation as a `build_fn`.
    """
    model = tf.keras.models.Sequential()
    model.add(Dense(units=n_input_nodes, activation=activation))
    for _ in range(n_hidden_layers):
        model.add(
            Dense(
                units=n_hidden_nodes,
                kernel_regularizer=L1L2(l1=l1, l2=l2),
                activation=activation,
            )
        )
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=n_output_nodes, activation=activation))

    model.compile(loss=loss, optimizer=optimizer, metrics=["mse"])

    return model
