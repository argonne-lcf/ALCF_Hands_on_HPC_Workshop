# Description

We assume that you have cloned the repo to a suitable location. These are the steps to execute this code on ThetaGPU (interactively):
1. Login to a ThetaGPU head node
```
ssh thetagpusn1
```
2. Request an interactive session on an A100 GPU
```
qsub -n 1 -q default -A datascience -I -t 1:00:00
```
Following this, we need to execute a few commands to get setup with an appropriately optimized tensorflow. These are:
3. Activate the TensorFlow 2.2 singularity container:
```
singularity exec -B /lus:/lus --nv /lus/theta-fs0/projects/datascience/thetaGPU/containers/tf2_20.08-py3.sif bash
```
4. Setup access to the internet
```
export HTTP_PROXY=http://theta-proxy.tmi.alcf.anl.gov:3128
export HTTPS_PROXY=https://theta-proxy.tmi.alcf.anl.gov:3128
```
Now that we can access the internet, we need to set up a virtual environment in Python (these commands should only be run the first time)
```
python -m pip install --user virtualenv
export VENV_LOCATION=/home/rmaulik/THETAGPU_TF_ENV # Add your path here
python -m virtualenv --system-site-packages $VENV_LOCATION
source $VENV_LOCATION/bin/activate
python -m pip install cmake
python -m pip install matplotlib
python -m pip install sklearn
```
5. Now we are ready to build our executable by executing the provided shell script (within the cloned repo):
```
source setup.sh
```
6. You can now run the app using
```
mpirun -n 1 -npernode 1 -hostfile $COBALT_NODEFILE ./app
```

