# Description

We assume that you have cloned the repo to a suitable location. These are the steps to execute this example on ThetaKNL from a script. We delay instructions on how to build the conda environment further in this file. In fact, the examples are set up to run with a pre-built environment that all attendees should have acces to.

1. Build the Fortran data loader and the SmartRedis Fortran client API.
- Move to the `example/src/` directory.
- Update the path to the `SmartRedis` directory made during the creation of the conda environment. Using the notation from the instructions below, this would be `/path/to/env/smartredis-0.2.0`.
- Set the environment in the ThetaKNL terminal with `source env_Theta.sh`.
- Build the code with `./doConfig.sh`.

2. Submit the job.
- Within the `example` directory, submit the job executing the script `./submit.sh`. This will launch the job in script mode.
- If you wish to submit an interactive job, simply execute the following command from the terminal.
```
qsub -I -q training-knl -n 4 -t 30 -A $SDL_Workshop
```
Then, once the interactive session starts, in order to run with the same parameters set by the submit script, execute the following from the MOM node
```
./run.sh 64 4 256 1 2 1 128 64
```

3. Monitor the job output and view the results of the prediction.
- When submitting in scropt mode, monitor the output of the `run.sh` and `driver.py` scripts with
```
tail -f JOBID.output
```
where JOBID is the job ID number assigned during submission. Then monitor the progress of the data loader with
```
tail -f load_data.out
```
and the progress of the training with
```
tail -f train_model.out
```
- Once the job has completed, those files can be viewed with any text editor and a comparison of the model predictions to the true target is available in a figure saved to `fig.pdf`.
- When running in interactive mode, the same files are available to view as the job is running.





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
