# Description

We assume that you have cloned the repo to a suitable location. These are the steps to execute this example on ThetaKNL from a script. We delay instructions on how to build the conda environment further in this file. In fact, the examples are set up to run with a pre-built environment that all attendees should have acces to.
The pre-build environment is specified at the top of the `run.sh` file [here](example/run.sh).

1. Build the Fortran data loader and the SmartRedis Fortran client API.
- Move to the `example/src/` directory.
- Set the environment in the ThetaKNL terminal with `source env_Theta.sh`.
- Build the code with `./doConfig.sh`.
- NOTE: When using your own conda environment rather then the pre-built one, the path to the SmartRedis client source files much be updated in the `CMakeLists.txt` file (lines 13, 18, 22, 23, 24).  

2. Submit the job.
- Within the `example` directory, submit the job executing the script 
```
./submit.sh
```
This will launch the job in script mode.
- If you wish to submit an interactive job, simply execute the following command from the terminal.
```
qsub -I -q training-knl -n 4 -t 30 -A SDL_Workshop
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

## Modify the job's parameters
The `submit.sh` script [here](example/submit.sh) defines the parameters of the job, such as the number of nodes and ranks used by the database, by the simulation and by the ML program. You can feel free to change those parameters and explore performance changes, however keep in mind the following details.
- When using more than 1 node for the database changing the value of `dbnodes`, a few lines of the source code need to change in order to initialize the clients to connect to a database cluster. Line 27 of the [data loader](example/src/load_data.f) must be changed to reflect the new size of the database. Similarly, line 35 of the [training](example/src/trainPar.py).
- A database cluster must request at least 3 nodes, meaning that one can't select to run the database on 2 nodes.
- `simprocs` is the number of processes the data loader runs with. In the example, a value of 128 was set because we set one process per core and used all 128 cores available on 2 nodes. In general, one does not have to use all cores on a node.
- `mlprocs` is the number of processes the data consumer runs with. In the example, a value of 64 was used to use all 64 cores on the node assigned to the ML program. This value can be increased or decreased along with the value of `mlnodes` to scale the training up or down.



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
