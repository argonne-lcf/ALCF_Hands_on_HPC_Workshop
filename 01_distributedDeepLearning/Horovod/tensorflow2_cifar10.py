# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import argparse
# Horovod: import Horovod module
import horovod.tensorflow as hvd
import time
t0 = time.time()
parser = argparse.ArgumentParser(description='TensorFlow CIFAR10 Example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 4)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--device', default='cpu',
                    help='Wheter this is running on cpu or gpu')
parser.add_argument('--num_inter', default=2, help='set number inter', type=int)
parser.add_argument('--num_intra', default=0, help='set number intra', type=int)

args = parser.parse_args()

    
# Horovod: initialize Horovod.
hvd.init()
print("I am rank %s of %s" %(hvd.rank(), hvd.size()))

# Horovod: pin GPU to be used to process local rank (one GPU per process)
if args.device == 'cpu':
    tf.config.threading.set_intra_op_parallelism_threads(args.num_intra)
    tf.config.threading.set_inter_op_parallelism_threads(args.num_inter)
else:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

(cifar10_images, cifar10_labels), _ = \
    tf.keras.datasets.cifar10.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(cifar10_images[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(cifar10_labels, tf.int64))
)
nsamples = len(list(dataset))
dataset = dataset.repeat().shuffle(1000).batch(args.batch_size)

loss = tf.losses.SparseCategoricalCrossentropy()

# Horovod: adjust learning rate based on number of GPUs.
opt = tf.optimizers.Adam(args.lr * hvd.size())
cifar10_model = tf.keras.applications.resnet50.ResNet50(include_top=True, input_tensor=None, input_shape=(32, 32, 3), pooling=None, classes=10, weights=None)
if (hvd.rank()==0):
    cifar10_model.summary()

checkpoint_dir = './checkpoints/tf2_cifar10'
checkpoint = tf.train.Checkpoint(model=cifar10_model, optimizer=opt)

@tf.function
def training_step(images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = cifar10_model(images, training=True)
        pred = tf.math.argmax(probs, axis=1)
        loss_value = loss(labels, probs)
        equality = tf.math.equal(pred, labels)
        accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape)
    grads = tape.gradient(loss_value, cifar10_model.trainable_variables)
    opt.apply_gradients(zip(grads, cifar10_model.trainable_variables))

    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if first_batch:
        hvd.broadcast_variables(cifar10_model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)
    return loss_value, accuracy


# Horovod: adjust number of steps based on number of GPUs.
nsteps = nsamples // hvd.size() // args.batch_size
for ep in range(args.epochs):
    running_loss = 0.0
    running_acc = 0.0
    tt0 = time.time()
    for batch, (images, labels) in enumerate(dataset.take(nsteps)):
        loss_value, acc  = training_step(images, labels, (batch == 0) and (ep == 0))
        running_acc += acc/nsteps
        running_loss += loss_value/nsteps
        if batch % 10 == 0:
            print('[%d] Epoch - %d, step #%d/%d\tLoss: %.6f' % (hvd.rank(), ep, batch, nsteps, loss_value))
        if hvd.rank() == 0 and batch % 10 == 0:
            checkpoint.save(checkpoint_dir)
    total_loss = hvd.allreduce(running_loss, average=True)
    total_acc = hvd.allreduce(running_acc, average=True)
    tt1 = time.time()
    if (hvd.rank()==0):
        print('Epoch - %d, \tTotal Loss: %.6f, \t Accuracy: %.6f, \t Time: %.6f seconds' % (ep, total_loss, total_acc, tt1 - tt0))

# Horovod: save checkpoints only on worker 0 to prevent other workers from
# corrupting it.
if hvd.rank() == 0:
    checkpoint.save(checkpoint_dir)
t1 = time.time()
if hvd.rank()==0:
    print("Total training time: %s seconds" %(t1 - t0))
