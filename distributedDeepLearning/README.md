# Distributed Deep Learning

Let by Huihuo Zheng <huihuo.zheng@anl.gov>; adopted some materials from  Corey Adams and Zhen Xie as well. 

This section of the workshop will introduce to you the methods we use to run distributed deep learning training on ALCF resources like ThetaGPU and Polaris. 

We show distributed training using three frameworks: 
1. [Horovod](Horovod/README.md) (for [TensorFlow](tensorflow.org) and [PyTorch](pytorch.org)), and
2. [DistributedDataParallel](DDP/README.md) (DDP) (for PyTorch only).
3. [DeepSpeed](DeepSpeed/README.md)