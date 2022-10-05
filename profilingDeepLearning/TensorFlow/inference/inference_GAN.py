import sys, os
import time

import logging
from logging import handlers

import tensorflow as tf
import numpy

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 "

def configure_logger():
    '''Configure a global logger

    Adds a stream handler and a file hander, buffers to file (10 lines) but not to stdout.

    Submit the MPI Rank

    '''
    logger = logging.getLogger()

    # Create a handler for STDOUT, but only on the root rank.
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    handler = handlers.MemoryHandler(capacity = 0, target=stream_handler)
    logger.addHandler(handler)

    # Add a file handler too:
    log_file =  "inference.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler = handlers.MemoryHandler(capacity=10, target=file_handler)
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)




# And here is our generator model, which will expect as input some random-number fixed length tensor.

class Generator(tf.keras.models.Model):

    def __init__(self, activation=tf.nn.tanh):
        tf.keras.models.Model.__init__(self)

        # The first step is to take the random image and use a dense layer to
        #make it the right shape

        self.dense = tf.keras.layers.Dense(
            units = 7 * 7 * 64
        )

        # This will get reshaped into a 7x7 image with 64 filters.

        # We need to upsample twice to get to a full 28x28 resolution image


        self.batch_norm_1 = tf.keras.layers.BatchNormalization()


        self.generator_layer_1 = tf.keras.layers.Convolution2D(
            kernel_size = [5, 5],
            filters     = 64,
            padding     = "same",
            use_bias    = True,
            activation  = activation,
        )

        self.unpool_1 = tf.keras.layers.UpSampling2D(
            size          = 2,
            interpolation = "nearest",
        )

        # After that unpooling the shape is [14, 14] with 64 filters
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()


        self.generator_layer_2 = tf.keras.layers.Convolution2D(
            kernel_size = [5, 5],
            filters     = 32,
            padding     = "same",
            use_bias    = True,
            activation  = activation,
        )

        self.unpool_2 = tf.keras.layers.UpSampling2D(
            size          = 2,
            interpolation = "nearest",
        )

        self.batch_norm_3 = tf.keras.layers.BatchNormalization()


        self.generator_layer_3 = tf.keras.layers.Convolution2D(
            kernel_size = [5, 5],
            filters     = 8,
            padding     = "same",
            use_bias    = True,
            activation  = activation,
        )

        # Now it is [28, 28] by 24 filters, use a bottle neck to
        # compress to a single image:
        self.batch_norm_4 = tf.keras.layers.BatchNormalization()


        self.generator_layer_final = tf.keras.layers.Convolution2D(
            kernel_size = [5, 5],
            filters     = 1,
            padding     = "same",
            use_bias    = True,
            activation  = tf.nn.sigmoid,
        )



    def call(self, inputs):
        '''
        Reshape at input and output:
        '''


        batch_size = inputs.shape[0]


        x = self.dense(inputs)


        # First Step is to to un-pool the encoded state into the right shape:
        x = tf.reshape(x, [batch_size, 7, 7, 64])

        x = self.batch_norm_1(x)
        x = self.generator_layer_1(x)
        x = self.unpool_1(x)
        x = self.batch_norm_2(x)
        x = self.generator_layer_2(x)
        x = self.unpool_2(x)
        x = self.batch_norm_3(x)
        x = self.generator_layer_3(x)
        x = self.batch_norm_4(x)
        x = self.generator_layer_final(x)


        return x




@tf.function
def generate_images(_generator, _batch_size):

    # Use the generator to make fake images:
    random_noise = tf.random.uniform(shape=[_batch_size,100], minval=-1, maxval=1, dtype=tf.float16)
    fake_images  = _generator(random_noise)

    return fake_images


def inference_GAN(_batch_size, _iterations):

    logger = logging.getLogger()

    generator = Generator()

    random_input = numpy.random.uniform(-1,1,[1,100]).astype(numpy.float16)
    generated_image = generator(random_input)

    # Load the model
    generator.load_weights("../trained_GAN_100epochs.h5")


    for i in range(_iterations):
        start = time.time()
        images = generate_images(generator, _batch_size)
        end = time.time()
        logger.info(f"Inference Images per second: {_batch_size / (end - start):.3f}")


    display_images = False
    if display_images:
        # Show an image:
        from matplotlib import pyplot as plt

        plot_images = numpy.zeros([4*28, 4*28])

        for i in range(4):
            for j in range(4):

                plot_images[i*28:(i+1)*28, j*28:(j+1)*28] = tf.reshape(images[i*4 + j], [28, 28])

        # image = tf.reshape(images[0], [28, 28])

        fig = plt.figure(figsize=(8,8))
        plt.imshow(plot_images)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':


    configure_logger()

    BATCH_SIZE=8192
    N_ITERATIONS = 200
    inference_GAN(BATCH_SIZE, N_ITERATIONS)
