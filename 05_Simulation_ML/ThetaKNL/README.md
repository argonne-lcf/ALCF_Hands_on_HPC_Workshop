# Description

These are the steps to execute this code on ThetaKNL (interactively):

1. Request an interactive session on Theta
```
qsub -n 1 -q debug-cache-quad -A datascience -I -t 1:00:00
```
2. Setup the Python environment to include TensorFlow and PyTorch (though we only need the former)
```
source setup.sh
```
3. If build is successful you are ready to run the example as follows
```
aprun -n 1 -N 1 -e OMP_NUM_THREADS=32 -d 32 -j 2 -e KMP_BLOCKTIME=0 -cc depth ./app
```

The same can be achieved by submitting to a queue using `qsub` by
```
source submit.sh
```