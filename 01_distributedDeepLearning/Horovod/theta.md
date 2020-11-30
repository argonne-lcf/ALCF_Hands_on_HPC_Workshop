# Hands on for Data Parallel Deep Learning on Theta (CPU)

1. Request an interactive session on Theta
```bash
qsub -n 4 -q debug-cache-quad -A datascience -I -t 1:00:00
```

3. Setup the Python environment to include TensorFlow, Keras, PyTorch and Horovod

For PyTorch
```bash
module load datascience/pytorch-1.4
```

For TensorFlow
```bash
module load datascience/tensorflow-2.2
```

4. Run examples
  -  PyTorch MNIST
  
```bash
aprun -n 16 -N 4 -e OMP_NUM_THREADS=32 -d 32 -j 2 -e KMP_BLOCKTIME=0 -cc depth python pytorch_mnist.py --device cpu
```

  -  TensorFlow MNIST
  
```bash
aprun -n 16 -N 4 -e OMP_NUM_THREADS=32 -d 32 -j 2 -e KMP_BLOCKTIME=0 -cc depth python tensorflow2_mnist.py --device cpu
```

  - TensorFlow Keras MNIST
  
```bash
aprun -n 16 -N 4 -e OMP_NUM_THREADS=32 -d 32 -j 2 -e KMP_BLOCKTIME=0 -cc depth python tensorflow2_keras_mnist.py --device cpu
```

5. Testing scaling and investigating the issue of large batch size training (this requires new queues )
You can do a simply scaling test. 
```bash
for n in 1 2 4 8 16 32 64 
do
  qsub -O pytorch_mnist_${n}nodes_t -n ${n} -q ATPESC2020 -A ATPESC2020 sumissions/theta/qsub_pytorch_mnist.sh
done
```
You can check the test accuracy and the timing for different scales. 

6. Warmup epochs
We could use a small learning rate (do not scale by the number of workers) in the begining 1 or 2 epochs, and see whether that improve the training results at large scale. 