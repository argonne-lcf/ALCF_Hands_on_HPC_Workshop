""""
model_run.py
"""
# pylint:disable=invalid-name, wrong-import-position
import os
import sys

import tensorflow as tf

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from load_data import load_data

keras = tf.keras
layers = tf.keras.layers

HISTORY = None

# pylint:disable=redefined-outer-name, too-many-locals
def run(point: dict = None):
    """Run the model at a "point" in hyperparameter space. Returns accuracy.

    point is expected to have the following keys:
     - units (list): containing the number of units in each hidden layer
     - activations (list): containing activations of each hidden layer
     - dropout_prob (float): dropout probability; if set to 0 no dropout layer used.
     - batch_size (int): batch size to use
     - learning_rate (float): Learning rate to use for training
     - optimizer (str): `keras.optimizers.Optimizer`
     - epochs (int): number of epochs
    """
    global HISTORY  # pylint:disable=global-statement
    if point is None:
        point = POINT

    print(point)

    (x_train, y_train), (x_test, y_test) = load_data()

    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    epochs = 10
    #  epochs = point.get('epochs', None)
    optimizer = point.get('optimizer', None)
    batch_size = point.get('batch_size', None)
    activation = point.get('activation', None)

    units1 = point.get('units1', None)
    units2 = point.get('units2', None)
    dropout1 = point.get('dropout1', None)
    dropout2 = point.get('dropout2', None)

    model = tf.keras.Sequential([
        layers.Dense(units1, activation=activation),
        layers.Dropout(dropout1),
        layers.Dense(units2, activation=activation),
        layers.Dropout(dropout2),
        layers.Dense(10)
    ])

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    _ = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        # We need to pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_val, y_val),
    )
    score = model.evaluate(x_test, y_test, verbose=2)
    print(f'test_loss, test_accuracy: {score[0]}, {score[1]}')

    return score[1]


if __name__ == "__main__":
    from problem import Problem

    point = Problem.starting_point_asdict[0]
    run(point)
