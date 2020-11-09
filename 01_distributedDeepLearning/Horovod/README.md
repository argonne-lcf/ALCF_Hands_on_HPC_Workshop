# Examples for Data parallel training 

Author: Huihuo Zheng

Purpose: ATPESC 2020 Machine Learning Track - Data Parallel Training

This folder contains some examples for data parallel training using Horovod. The examples are adapted from Horovod [github](https://github.com/horovod/horovod/tree/master/examples) with some modifications. 

## Setup the environement
setup.sh

## List of Examples 

* tensorflow2_mnist.py  - TF2 MNIST example
* tensorflow2_keras_mnist.py - TF2 Keras MNIST example
* pytorch_mnist.py - PyTorch MNIST example

These examples were created based on https://github.com/horovod/horovod/tree/master/examples. The original examples from Horovod are assumed to be run on GPU. I modified them to be able to run on CPUs.

## Submission scripts:
   The submission scripts are for Theta @ ALCF: submissions/theta/qsub_*.sh

## Running the examples
```bash
  qsub -A ATPESC2020 -q ATPESC2020 -t 1:00:00 -n 4 submissions/theta/qsub_keras_mnist.py
```
* modify ```-n``` and ```-t``` to run on different number of nodes with different walltime. 
* We assume one worker per node in all the cases. Modify PROC_PER_NODE and num_intra, num_threads accordingly if you want to set more than one workers per node. 