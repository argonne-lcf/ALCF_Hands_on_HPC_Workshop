# Introduction to Distributed Training

This section of the workshop will introduce to you the methods we use to run distributed deep learning training on ALCF resources like Theta and ThetaGPU.

We show distributed training using two frameworks: 
1. [Horovod](Horovod/) (for [TensorFlow](tensorflow.org) and [PyTorch](pytorch.org)), and
2. [DistributedDataParallel](DDP/) (DDP) (for PyTorch only).

Some instructions on Data Parallel Training can be found here:

https://argonne-lcf.github.io/ThetaGPU-Docs/data_parallel_training/