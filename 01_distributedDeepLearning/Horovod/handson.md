# Hands on for Data Parallel Deep Learning

1. Copy the folder to your home directory
```bash 
cp /gpfs/mira-home/hzheng/projects/ATPESC2020/hzheng/ATPESC_MachineLearning/DataParallelDeepLearning ~/DataParallelDeepLearning
[ -e ~/.keras ] || mkdir ~/.keras
[ -e ~/.keras/datasets ] || mkdir ~/.keras/datasets
cp /gpfs/mira-home/hzheng/projects/ATPESC2020/hzheng/ATPESC_MachineLearning/DataParallelDeepLearning/datasets/mnist.npz ~/.keras/datasets
```

2. Request an interactive session on Theta
```bash
qsub -n 4 -q ATPESC2020 -A ATPESC2020 -I -t 1:00:00
```

3. Setup the Python environment to include TensorFlow, Keras, PyTorch and Horovod

```bash
cd ~/DataParallelDeepLearning
source setup.sh
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
aprun -n 16 -N 4 -e OMP_NUM_THREADS=32 -d 32 -j 2 -e KMP_BLOCKTIME=0 -cc depth python tensorflow2_keras_mnist.py  --device cpu
```

5. Testing scaling and investigating the issue of large batch size training
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