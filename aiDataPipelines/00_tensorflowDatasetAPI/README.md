# Tensorflow Example Data Pipeline
by J. Taylor Childers (jchilders@anl.gov)

This is the example impliments a data pipeline for the ImageNet dataset described in the [README](../README.md) one level up.

Example submission script for Polaris is provided.

Submit to Polaris using:
```bash
qsub -A <project> -q <queue> submit_polaris.sh
```

All log files go into the `logdir` folder.


# Profiler View

You can view the processes and how they occupy the compute resources in Tensorflow using tensorboard.

You can login to Polaris using:
```bash
# our proxy port, must be > 1024
export PORT=10001
# login to theta with a port forwarding
ssh -D $PORT user@polaris.alcf.anl.gov
# load any conda environment that has a compatible tensorboard installation
module use /soft/modulefiles
module load conda
conda activate
cd <path/to/dataPipelines/00_tensorflowDatasetAPI/>
# start tensorboard (load_fast==false is a recent setting that seems to be needed until Tensorflow work's out the bugs)
tensorboard --bind_all --logdir logdir
```
Note the Port number that `tensorboard` reports when it starts up.

Only 1 user can use a specific port so if you get an error choose another port number larger than `1024`.

Once you have that setup. Set the Socks5 proxy of your favorite browser to host `localhost` and port `$PORT` (where `$PORT` is the value you used in the above script, like `10001`). Now in the browser URL enter the login node on which you started `tensorboard`. For instance, now you can type in `localhost:6006`. Here `6006` is the port that `tensorboard` uses by default to start up it's web service, but may vary if that port is already used so note the port reported by `tensorboard` when it starts.
