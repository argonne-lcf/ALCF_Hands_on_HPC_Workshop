# Hands on for Data Parallel Deep Learning on ThetaGPU

1. Request an interactive session on Theta:
```bash
qsub -n 1 -q full-node -A datascience -I -t 1:00:00
```

2. Setup the Python environment to include TensorFlow, Keras, PyTorch, and Horovod:
```bash
source /lus/theta-fs0/software/datascience/thetagpu/anaconda3/setup.sh
```

3. Run examples
   - PyTorch MNIST - 8 GPUs
     ```bash
     mpirun -np 8 python pytorch_mnist.py --device gpu
     ```

   - PyTorch CIFAR10 - 8 GPUs
     ```bash
     mpirun -np 8 python pytorch_cifar10.py --device gpu
     ```

   -  TensorFlow MNIST
      ```bash
      mpirun -np 8 python tensorflow2_mnist.py --device gpu
      ```

   - TensorFlow Keras MNIST
     ```bash
     mpirun -np 8 python  tensorflow2_keras_mnist.py --device gpu
     ```
     
4. Test scaling and investigate the issue of large batch size training

    Note, this requires a new job allocation to a separate job queue. The following script performes a simple scaling test with the MNIST dataset and a PyTorch model:
   ```bash
   for n in 1 2 4 8
   do
     	mpirun -np $n python tensorflow2_keras_mnist.py --device gpu >& tensorflow2_keras_mnist.out
   done
   ```
   You can check the test accuracy and the timing for different scales.

5. Try using warmup epochs

    We could use a small learning rate (do not scale by the number of workers) in the begining 1 or 2 epochs, and see whether that improve the training results at large scale.
