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

OPTIMIZERS = {
    'Adam': tf.keras.optimizers.Adam,
    'Nadam': tf.keras.optimizers.Nadam,
    'Adagrad': tf.keras.optimizers.Adagrad,
    'RMSprop': tf.keras.optimizers.RMSprop,
}


def get_optimizer(
        opt_id: str,
        learning_rate: float = None,
        momentum: float = None,
):
    """Returns the optimizer conditioned on `opt_id`."""
    if opt_id == 'SGD':
       return tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                      momentum=momentum)

    return OPTIMIZERS[str(opt_id)](learning_rate=learning_rate)


# pylint:disable=redefined-outer-name, too-many-locals
def run(point: dict):
    """Run the model at a point in hyperparameter space.

    `point` is expected to have the following keys:
        - units1 (int): Number of units in first hidden layer
        - units2 (int): Number of units in second hidden layer
        - activation (str): Activation function to use
        - dropout_prob (float): Dropout probability; if set to 0 no dropout
        - batch_size (int): Batch size to use
        - log10_learning_rate (float): log(Learning rate) to use for training
        - optimizer (str): ['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Nadam']
        - momentum (float): Momentum to use; ONLY for SGD optimizer

    Returns:
        - accuracy (float): The real-valued objective to be maximized.
    """
    (x_train, y_train), (x_test, y_test) = load_data()

    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    epochs = 10
    # Parse hyperparameters from `point`
    batch_size = point.get('batch_size', None)
    activation = point.get('activation', None)

    units1 = point.get('units1', None)
    units2 = point.get('units2', None)
    dropout1 = point.get('dropout1', None)
    dropout2 = point.get('dropout2', None)

    opt_id = point.get('optimizer', None)
    momentum = point.get('momentum', None)
    log10_learning_rate = point.get('log10_learning_rate', None)
    learning_rate = None
    if log10_learning_rate is not None:
        learning_rate = 10.0**log10_learning_rate

    # the 'optimizer' is stored as a string in `point`,
    # we can get the actual optimizer using the `get_optimizer` fn
    optimizer = get_optimizer(opt_id=opt_id,
                              learning_rate=learning_rate,
                              momentum=momentum)

    # Build the model
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
