# Distributed training with Horovod

Author: Huihuo Zheng <huihuo.zheng@anl.gov>

We provide MNIST and CIFAR10 examples for data parallel training. These examples were adopted from Horovod with modification. 

* PyTorch example
  * `pytorch_mnist.py`
  * `pytorch_cifar10.py`
* TensorFlow example
  * `tensorflow2_mnist.py`
  * `tensorflow2_cifar10.py`
* Keras (TensorFlow) example
  * `tensorflow2_keras_mnist.py`
  * `tensorflow2_keras_cifar10.py`
  
All the examples can be run either on CPUs or GPUs by specifying ```--device [cpu|gpu]```. 

Please check `theta.md` and `thetagpu.md` for instructions on how to run the examples on Theta and ThetaGPU, respectively. 

For submitting jobs in the script (non-interactive) job mode, consult the submission scripts in the `./submissions/` folder. 
