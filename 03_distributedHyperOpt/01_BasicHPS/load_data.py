"""
load_data.py
"""
import tensorflow as tf


def load_data():
    """Returns mnist data as `(x_train, y_train), (x_test, y_test)`."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

    print(f'(x_train, y_train): ({x_train.shape}, {y_train.shape})')
    print(f'(x_test, y_test): ({x_test.shape}, {y_test.shape})')

    return ((x_train, y_train), (x_test, y_test))


if __name__ == '__main__':
    load_data()
