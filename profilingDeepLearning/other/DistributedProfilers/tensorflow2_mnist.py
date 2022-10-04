import tensorflow as tf
import argparse
import horovod.tensorflow as hvd
import time


def init_args():
    parser = argparse.ArgumentParser(description='TensorFlow MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--device', default='cpu',
                        help='Wheter this is running on cpu or gpu')
    parser.add_argument('--num_inter', default=2, help='set number inter',
                        type=int)
    parser.add_argument('--num_intra', default=0, help='set number intra',
                        type=int)
    args = parser.parse_args()
    return args


def init_hvd(args):
    # Horovod: initialize Horovod.
    hvd.init()
    if args.device == 'gpu':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()],
                                                       'GPU')


def init_dataset(batch_size):
    (mnist_images, mnist_labels), _ = \
        tf.keras.datasets.mnist.load_data(path='mnist.npz')

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
            tf.cast(mnist_labels, tf.int64)))
    dataset = dataset.shard(hvd.size(), hvd.rank()) \
                     .shuffle(10**4, reshuffle_each_iteration=True) \
                     .batch(batch_size) \
                     .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == '__main__':
    args = init_args()
    init_hvd(args)
    dataset = init_dataset(args.batch_size)

    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    loss = tf.losses.SparseCategoricalCrossentropy()

    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.optimizers.Adam(args.lr * hvd.size())

    @tf.function
    def training_step(images, labels, first_batch):
        with tf.GradientTape() as tape:
            probs = mnist_model(images, training=True)
            loss_value = loss(labels, probs)

        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss_value, mnist_model.trainable_variables)
        opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

        # Horovod: broadcast initial variable states from rank 0 to all other
        # processes. This is necessary to ensure consistent initialization of
        # all workers when training is started with random weights or restored
        # from a checkpoint.
        #
        # Note: broadcast should be done after the first gradient step to
        # ensure optimizer initialization.
        # print("PREPARE")
        if first_batch:
            hvd.broadcast_variables(mnist_model.variables, root_rank=0)
            hvd.broadcast_variables(opt.variables(), root_rank=0)

        return loss_value

    ttime = 0
    n_warmup = 2
    for ne in range(args.epochs):
        start = time.time()
        epoch_loss = 0.
        for batch, (images, labels) in enumerate(dataset):
            epoch_loss += training_step(images, labels,
                                        (batch == 0) and (ne == 0))
        end = time.time()
        timer = end - start

        if ne >= n_warmup:
            ttime += timer
        if hvd.rank() == 0:
            print(f'Epoch: {ne} \tLoss: {epoch_loss:0.3f} \t'
                  f'Time/epoch: {timer:.3g}')

    ttime /= (args.epochs - n_warmup)
    if hvd.rank() == 0:
        print(f"Average epoch training time: {ttime:.3g} seconds")
