# PyTorch Example Data Pipeline
by J. Taylor Childers (jchilders@anl.gov)

This is very similar to the Tensorflow example covered in the [README](../README.md) one level up.

Example submission scripts are provided. 

Be aware that as of Sept 28 2022 there is an issue with PyTorch + MPI + multi-worker data APIs on Polaris. We are working to understand the issue and fix it.

Submit to Polaris using:
```bash
qsub -A <project> -q <queue> submit_polaris.sh
```

Submit to ThetaGPU using:
```bash
qsub -A <project> -q <queue> submit_thetagpu.sh
```

Submit to ThetaKNL using:
```bash
qsub -A <project> -q <queue> submit_thetknl.sh
```

All log files go into the `logdir` folder.
