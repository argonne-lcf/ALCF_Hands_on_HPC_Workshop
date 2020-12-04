# Hands on for Data Parallel Deep Learning on ThetaGPU

1. Request an interactive session on ThetaGPU:
```bash
# Login to theta
ssh -CY user@theta.alcf.anl.gov 
# Login to ThetaGPU login node
ssh -CY thetagpusn1 
# Requesting 1 node 
qsub -n 1 -q full-node -A SDL_Workshop -I -t 2:00:00
```

2. Setup the Python environment to include TensorFlow, Keras, PyTorch, and Horovod:
	- **For PyTorch**
```bash
source /lus/theta-fs0/projects/datascience/parton/thetagpu/pt-build/pt-intall/mconda3/setup.sh
```
	- **For TensorFlow**
```bash
source /lus/theta-fs0/software/thetagpu/conda/tf_master/latest/mconda3/setup.sh
```

3. Run examples on a single node
   - PyTorch MNIST - 8 GPUs
     ```bash
     mpirun -np 8 python pytorch_mnist.py --device gpu
     ```

   - PyTorch CIFAR10 - 8 GPUs
     ```bash
     mpirun -np 8 python pytorch_cifar10.py --device gpu
     ```

   -  TensorFlow MNIST - 8 GPUs
      ```bash
      mpirun -np 8 python tensorflow2_mnist.py --device gpu
      ```

   - TensorFlow Keras MNIST - 8 GPUs
     ```bash
     mpirun -np 8 python  tensorflow2_keras_mnist.py --device gpu
     ```
     
4. Test scaling and investigate the issue of large batch size training

   The following script performes a simple scaling test with the MNIST dataset and a PyTorch model:
   ```bash
   for n in 1 2 4 8
   do
     	mpirun -np $n python tensorflow2_keras_mnist.py --device gpu >& tensorflow2_keras_mnist.out.$n
   done
   ```
   If you want to go to larger scale (16 GPUs), you could run on 2 nodes. 
  ```bash
  mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np 16 -npernode 8 --hostfile $COBALT_NODEFILE  python tensorflow2_keras_mnist.py --device gpu >& tensorflow2_keras_mnist.out.16
  ```
   You can check the test accuracy and the timing for different scales.
  
   We prepare some submission script in ./submissions/thetagpu/qsub_*
   
   
   
   
5. Try using warmup epochs

    We could use a small learning rate (do not scale by the number of workers) in the begining 1 or 2 epochs, and see whether that improve the training results at large scale.
