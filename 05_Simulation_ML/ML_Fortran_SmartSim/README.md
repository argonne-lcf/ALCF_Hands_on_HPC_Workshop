# Description

We assume that you have cloned the repo to a suitable location. These are the steps to execute this example on ThetaKNL from a script. We delay instructions on how to build the conda environment further in this file.





# Building the Conda Environment

The commands required to build the conda environment are included in two scripts. The [first script](install_ssim_theta.sh) creates the new conda environment and installs SmartSim and SmartRedis. The [second script](install_horovod_theta.sh) installs Horovod in the environment.

To run the first script, execute
```
source install_ssim_theta.sh /path/to/env/
```
Note that the script takes as argument the path to the location where you wish this environment to be created. The name of the environment is `ssim` by default, but it can be changed within the script.
Also, please note that in some occasions, the last command in the script fails to execute successfully. If that is the case, you will notice an error message in the output and you simply execute the `make lib` command again to complete the build of SmartRedis.

To run the second script, make sure you are not within one of the `smartsim` or `smartredis` directories, and then execute
```
./install_horovod_theta.sh
```

Finally, to activate the environment, make sure to either execute these commands or include them in your scripts before running jobs with SmartSim.
```
module swap PrgEnv-intel PrgEnv-gnu
export CRAYPE_LINK_TYPE=dynamic
module load miniconda-3/2021-07-28
conda activate /path/to/env/ssim
```
