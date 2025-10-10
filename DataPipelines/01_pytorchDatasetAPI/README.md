# PyTorch Example Data Pipeline
by J. Taylor Childers (jchilders@anl.gov)

This tutorial demonstrates how to build a parallel data pipeline in PyTorch using the Python `threading` module. The dataset used is the ImageNet dataset described in the [README](../README.md) one level up. The script uses [PyTorch Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) to distribute the training task across multiple nodes. We do not cover how DDP works in this tutorial.

Submit to Polaris using:
```bash
qsub -A alcf_training -q <queue> -v NUM_WORKERS=1 submit_polaris.sh
qsub -A alcf_training -q <queue> -v NUM_WORKERS=2 submit_polaris.sh
qsub -A alcf_training -q <queue> -v NUM_WORKERS=4 submit_polaris.sh
qsub -A alcf_training -q <queue> -v NUM_WORKERS=8 submit_polaris.sh
```

run_1.log:imgs/sec = 592.25
run_2.log:imgs/sec = 122.54
run_4.log:imgs/sec = 1560.03
run_8.log:imgs/sec = 1953.06
run_16.log:imgs/sec = 1934.75

