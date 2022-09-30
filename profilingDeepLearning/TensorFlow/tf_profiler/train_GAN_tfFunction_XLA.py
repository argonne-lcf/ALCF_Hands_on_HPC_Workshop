import sys, os
import time

os.environ["TF_XLA_FLAGS"]="--tf_xla_auto_jit=2"

import logging
from logging import handlers

import tensorflow as tf
import numpy

import horovod.tensorflow as hvd

"""
Training a Generative Adversarial Network

You can build a generative model using an adversarial training technique. This has been around for some time now, but the key components of the training process are this:

    Two networks are created.
        One is a generator, which is tasked with taking random input (Let's say we give it a tensor of shape 50, with each value in the range (0, 1) ) and outputing an image (shape 28x28 as usual) which is indistinguishable from the handwritten digits in the mnist dataset.
        The second network is a discriminator or classifier which takes as input images and has to decide on a case-by-case basis which images are from the real dataset, and which images are from the generator.
    During training, first the generator is run and it produces a set of output images.
    Next, a set of real images is pulled from the dataset and augments the generated images. Each image is given a label of 1 for fake, 0 for real.
    The discriminator runs and classifies the images as best it can, either real or fake.
    The gradients of the discriminator are updated easily, as it is just a classification problem.
    The gradients of the generator are then updated based on the success or failure of the discriminator. In this way, the generator is trained directly to trick the discriminator into believing the wrong images are real.

At the end of the training session, the discriminator network can usually be discarded while the generator remains as an interesting network. You can use it moving forward to generate new fake data.
"""

# Read in the mnist data so we have it loaded globally:
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(numpy.float32)

x_train /= 255.

x_train = tf.convert_to_tensor(x_train)

x_train = tf.random.shuffle(x_train)

dataset = tf.data.Dataset.from_tensor_slices((x_train))
dataset.shuffle(60000)



def init_mpi():

    # Using the presence of an env variable to determine if we're using MPI:


    try:
        hvd.init()
        local_rank = hvd.local_rank()
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

        return hvd.rank(), hvd.size()
    except:
        if "mpirun" in sys.argv or "mpiexec" in sys.argv:
            raise Exception("MPI detected in command line but was not able to init!")
        return 0, 1


def configure_logger(rank):
    '''Configure a global logger

    Adds a stream handler and a file hander, buffers to file (10 lines) but not to stdout.

    Submit the MPI Rank

    '''
    logger = logging.getLogger()

    # Create a handler for STDOUT, but only on the root rank.
    # If not distributed, we still get 0 passed in here.
    if rank == 0:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        handler = handlers.MemoryHandler(capacity = 0, target=stream_handler)
        logger.addHandler(handler)

        # Add a file handler too:
        log_file =  "process.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler = handlers.MemoryHandler(capacity=10, target=file_handler)
        logger.addHandler(file_handler)

        logger.setLevel(logging.INFO)
    else:
        # in this case, MPI is available but it's not rank 0
        # create a null handler
        handler = logging.NullHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


# The network designs for a GAN need not be anything special.

class Discriminator(tf.keras.models.Model):
    '''
    Simple classifier for mnist, but only 1 or 2 outputs:

    With 1 output, we will use a sigmoid cross entropy.

    '''

    def __init__(self, activation=tf.nn.tanh):
        tf.keras.models.Model.__init__(self)

        # Apply a 5x5 kernel to the image:
        self.discriminator_layer_1 = tf.keras.layers.Convolution2D(
            kernel_size = [5, 5],
            filters     = 24,
            padding     = "same",
            activation  = activation,
        )

        self.dropout_1 = tf.keras.layers.Dropout(0.4)

        self.pool_1 = tf.keras.layers.MaxPool2D()

        self.discriminator_layer_2 = tf.keras.layers.Convolution2D(
            kernel_size = [5, 5],
            filters     = 64,
            padding     = "same",
            activation  = activation,
        )

        self.dropout_2 = tf.keras.layers.Dropout(0.4)

        self.pool_2 = tf.keras.layers.MaxPool2D()

        self.discriminator_layer_3 = tf.keras.layers.Convolution2D(
            kernel_size = [5, 5],
            filters     = 128,
            padding     = "same",
            activation  = activation,
        )

        self.dropout_3 = tf.keras.layers.Dropout(0.4)


        self.pool_3 = tf.keras.layers.MaxPool2D()

        self.discriminator_layer_final = tf.keras.layers.Dense(
            units = 1,
            activation = None,
            )





    @tf.function
    def call(self, inputs):

        batch_size = inputs.shape[0]
        x = inputs
        # Make sure the input is the right shape:
        x = tf.reshape(x, [batch_size, 28, 28, 1])

        x = self.discriminator_layer_1(x)
        x = self.pool_1(x)
        x = self.dropout_1(x)
        x = self.discriminator_layer_2(x)
        x = self.pool_2(x)
        x = self.dropout_2(x)
        x = self.discriminator_layer_3(x)
        x = self.pool_3(x)
        x = self.dropout_3(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.discriminator_layer_final(x)

        return x





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


    @tf.function
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
def compute_loss(_logits, _targets):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=_targets, logits=_logits)

    return tf.reduce_mean(loss)


@tf.function
def forward_pass(_generator, _discriminator, _real_batch, _input_size):
        '''
        This function takes the two models and runs a forward pass to the computation of the loss functions
        '''

        real_data = _real_batch

        _batch_size = _real_batch.shape[0]

        # Use the generator to make fake images:

        # Use the generator to make fake images:
        random_noise = tf.random.uniform(shape=[_batch_size,_input_size], minval=-1, maxval=1)
        fake_images  = _generator(random_noise)


        # Use the discriminator to make a prediction on the REAL data:
        prediction_on_real_data = _discriminator(real_data)
        # Use the discriminator to make a prediction on the FAKE data:
        prediction_on_fake_data = _discriminator(fake_images)


        soften = 0.1
        real_labels = tf.zeros(shape=[_batch_size,1], dtype=tf.float32) + soften
        fake_labels = tf.ones( shape=[_batch_size,1], dtype=tf.float32) - soften
        gen_labels  = tf.zeros(shape=[_batch_size,1], dtype=tf.float32)



        # Occasionally, we disrupt the discriminator (since it has an easier job)

        # Invert a few of the discriminator labels:

        n_swap = int(_batch_size * 0.1)

        indices = tf.reshape(tf.range(n_swap), [-1,1])

        swap_real = tf.constant(-soften, shape = indices.shape)
        swap_fake = tf.constant( soften, shape = indices.shape)

        real_labels = real_labels + tf.scatter_nd(indices=indices, updates=swap_real, shape=real_labels.shape)
        fake_labels = fake_labels + tf.scatter_nd(indices=indices, updates=swap_fake, shape=fake_labels.shape)


        # Compute the loss for the discriminator on the real images:
        discriminator_real_loss = compute_loss(
            _logits  = prediction_on_real_data,
            _targets = real_labels)

        # Compute the loss for the discriminator on the fakse images:
        discriminator_fake_loss = compute_loss(
            _logits  = prediction_on_fake_data,
            _targets = fake_labels)

        # The generator loss is based on the output of the discriminator.
        # It wants the discriminator to pick the fake data as real
        generator_target_labels = [1] * _batch_size

        generator_loss = compute_loss(
            _logits  = prediction_on_fake_data,
            _targets = real_labels)

        # Average the discriminator loss:
        discriminator_loss = 0.5*(discriminator_fake_loss  + discriminator_real_loss)

        loss = {
            "discriminator" : discriminator_loss,
            "generator"    : generator_loss
        }

        return loss


# Here is a function that will manage the training loop for us:

def train_loop(batch_size, n_training_epochs, models, opts, global_size):

    @tf.function()
    def train_iteration(data, _models, _opts, _global_size):

        #Update the generator:
        with tf.GradientTape() as tape:
                loss = forward_pass(
                    models["generator"],
                    models["discriminator"],
                    _input_size = 100,
                    _real_batch = data,
                )


        if global_size != 1:
            tape = hvd.DistributedGradientTape(tape)




        trainable_vars = _models["generator"].trainable_variables

        # Apply the update to the network (one at a time):
        grads = tape.gradient(loss["generator"], trainable_vars)

        _opts["generator"].apply_gradients(zip(grads, trainable_vars))

        #Update the discriminator:
        with tf.GradientTape() as tape:
                loss = forward_pass(
                    models["generator"],
                    models["discriminator"],
                    _input_size = 100,
                    _real_batch = data,
                )


        if _global_size != 1:
            tape = hvd.DistributedGradientTape(tape)




        trainable_vars = _models["discriminator"].trainable_variables

        # Apply the update to the network (one at a time):
        grads = tape.gradient(loss["discriminator"], trainable_vars)

        _opts["discriminator"].apply_gradients(zip(grads, trainable_vars))


        return loss



    logger = logging.getLogger()


    rank = hvd.rank()
    tf.profiler.experimental.start('logdir')
    for i_epoch in range(n_training_epochs):

        epoch_steps = int(60000/batch_size)
        dataset.shuffle(60000) # Shuffle the whole dataset in memory
        batches = dataset.batch(batch_size=batch_size, drop_remainder=True)

        for i_batch, batch in enumerate(batches):

            data = tf.reshape(batch, [-1, 28, 28, 1])

            start = time.time()

            loss = train_iteration(data, models, opts, global_size)

            if loss["discriminator"] < 0.01:
                break


            end = time.time()

            images = batch_size*2*global_size


            logger.info(f"({i_epoch}, {i_batch}), G Loss: {loss['generator']:.3f}, D Loss: {loss['discriminator']:.3f}, step_time: {end-start :.3f}, throughput: {images/(end-start):.3f} img/s.")

    tf.profiler.experimental.stop()

# @tf.function
def train_GAN(_batch_size, _training_epochs, global_size):



    generator = Generator()

    random_input = numpy.random.uniform(-1,1,[1,100]).astype(numpy.float32)
    generated_image = generator(random_input)


    discriminator = Discriminator()
    classification  = discriminator(generated_image)

    models = {
        "generator" : generator,
        "discriminator" : discriminator
    }

    opts = {
        "generator" : tf.keras.optimizers.Adam(0.001),
        "discriminator" : tf.keras.optimizers.RMSprop(0.0001)


    }

    if global_size != 1:
        hvd.broadcast_variables(generator.variables, root_rank=0)
        hvd.broadcast_variables(discriminator.variables, root_rank=0)
        hvd.broadcast_variables(opts['generator'].variables(), root_rank=0)
        hvd.broadcast_variables(opts['discriminator'].variables(), root_rank=0)

    train_loop(_batch_size, _training_epochs, models, opts, global_size)

if __name__ == '__main__':

    rank, size = init_mpi()
    configure_logger(rank)

    BATCH_SIZE=1024
    N_TRAINING_EPOCHS = 10
    train_GAN(BATCH_SIZE, N_TRAINING_EPOCHS, size)
