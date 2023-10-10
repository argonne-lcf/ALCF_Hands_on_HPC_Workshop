# Horovod example
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MPICH_GPU_SUPPORT_ENABLED']='0'

import tensorflow as tf

import numpy
import time

#HVD: (1) Import horovod
import horovod.tensorflow as hvd
hvd.init()
print("I am rank %d of %d"%(hvd.rank(), hvd.size()))

import argparse
parser = argparse.ArgumentParser(description='TensorFlow MNIST Example')
parser.add_argument('--epochs', default=32,
                    type=int, help='Number of epochs to run')

parser.add_argument('--device', default='gpu',
                    help='Wheter this is running on cpu or gpu')
parser.add_argument('--num_inter', default=2, help='set number inter', type=int)
parser.add_argument('--num_intra', default=0, help='set number intra', type=int)
parser.add_argument('--batch_size', default=512, type=int)
args = parser.parse_args()
tf.config.threading.set_intra_op_parallelism_threads(args.num_intra)
tf.config.threading.set_inter_op_parallelism_threads(args.num_inter)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#HVD: (2) Pin GPU        
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# MNIST dataset 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(numpy.float32)
x_test  = x_test.astype(numpy.float32)

x_train /= 255.
x_test  /= 255.

y_train = y_train.astype(numpy.int32)
y_test  = y_test.astype(numpy.int32)

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#HVD: sharding the dataset
dataset = dataset.shard(hvd.size(), hvd.rank())
dataset = dataset.shuffle(60000)
batches = dataset.batch(batch_size = args.batch_size, drop_remainder=True)

# Convolutional model

class MNISTClassifier(tf.keras.models.Model):

    def __init__(self, activation=tf.nn.tanh):
        tf.keras.models.Model.__init__(self)

        self.conv_1 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu')
        self.conv_2 = tf.keras.layers.Conv2D(64, [3, 3], activation='relu')
        self.pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.drop_4 = tf.keras.layers.Dropout(0.25)
        self.dense_5 = tf.keras.layers.Dense(128, activation='relu')
        self.drop_6 = tf.keras.layers.Dropout(0.5)
        self.dense_7 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):

        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.pool_3(x)
        x = self.drop_4(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense_5(x)
        x = self.drop_6(x)
        x = self.dense_7(x)

        return x

def compute_loss(y_true, y_pred):
    # if labels are integers, use sparse categorical crossentropy
    # network's final layer is softmax, so from_logtis=False
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # if labels are one-hot encoded, use standard crossentropy

    return scce(y_true, y_pred)


def forward_pass(model, batch_data, y_true):
    y_pred = model(batch_data)
    loss = compute_loss(y_true, y_pred)
    return loss


def train_loop(batch_size, n_training_epochs, model, opt):
    
    @tf.function()
    def train_iteration(data, y_true, model, opt):
        with tf.GradientTape() as tape:
            loss = forward_pass(model, data, y_true)

        trainable_vars = model.trainable_variables
        #HVD: (4) distributed tape
        tape = hvd.DistributedGradientTape(tape)

        # Apply the update to the network (one at a time):
        grads = tape.gradient(loss, trainable_vars)

        opt.apply_gradients(zip(grads, trainable_vars))
        return loss
    
    for i_epoch in range(n_training_epochs):
        if (hvd.rank==0):
            print("beginning epoch %d" % i_epoch)
        start = time.time()
        total_loss = 0.0
        for i_batch, (batch_data, y_true) in enumerate(batches):
            batch_data = tf.reshape(batch_data, [-1, 28, 28, 1])
            if (args.device=='cpu'):
                with tf.device("/cpu:0"):
                    loss = train_iteration(batch_data, y_true, model, opt)
            else:
                loss = train_iteration(batch_data, y_true, model, opt)
            total_loss += loss
            #HVD: (5) broadcast from 0 (need to be done after first step to ensured that optimizer is initialized)
            if (i_batch==0 and i_epoch==0):
                hvd.broadcast_variables(model.variables, root_rank=0)
                hvd.broadcast_variables(opt.variables(), root_rank=0)
        #HVD: (6) average metrics
        total_loss = hvd.allreduce(total_loss, average=False)
        end = time.time()
        if (hvd.rank()==0):
            print("took %4.4f seconds for epoch #%d - %s" % (end-start, i_epoch, tf.print("loss: ", total_loss)))


def train_network(_batch_size, _n_training_epochs, _lr):

    mnist_model = MNISTClassifier()
    #HVD: (3) scale learning rate
    opt = tf.keras.optimizers.Adam(_lr*hvd.size())

    train_loop(_batch_size, _n_training_epochs, mnist_model, opt)


batch_size = 512
epochs = args.epochs
lr = .01
train_network(batch_size, epochs, lr)
