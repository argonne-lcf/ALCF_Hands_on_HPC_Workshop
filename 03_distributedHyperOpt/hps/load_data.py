import os
import numpy as np
import tensorflow as tf

def load_data(split: int=10000):
    """Generate a random distribution of data for polynome_2 function: -SUM(X**2) where "**" is an element wise operator in the continuous range [a, b].

    Args:
        dim (int): size of input vector for the polynome_2 function.
        a (int): minimum bound for all X dimensions.
        b (int): maximum bound for all X dimensions.
        prop (float): a value between [0., 1.] indicating how to split data between training set and validation set. `prop` corresponds to the ratio of data in training set. `1.-prop` corresponds to the amount of data in validation set.
        size (int): amount of data to generate. It is equal to `len(training_data)+len(validation_data).

    Returns:
        tuple(tuple(ndarray, ndarray), tuple(ndarray, ndarray)): of Numpy arrays: `(train_X, train_y), (valid_X, valid_y)`.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, 784))
    x_test = np.reshape(x_test, (-1, 784))

    x_val = x_train[-split:]
    y_val = y_train[-split:]
    x_train = x_train[:-split]
    y_train = y_train[:-split]

    print(f'x_train shape: {np.shape(x_train)}')
    print(f'y_train shape: {np.shape(y_train)}')
    print(f'x_val shape: {np.shape(x_val)}')
    print(f'y_val shape: {np.shape(y_val)}')

    return ((x_train, y_train), (x_val, y_val))


if __name__ == '__main__':
    load_data()
