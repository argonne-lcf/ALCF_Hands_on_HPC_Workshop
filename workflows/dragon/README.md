# Dragon HPC

Dragon HPC is a composable distributed run-time for managing processes, memory, and data at scale through high-performance communication objects.

This demo focuses on the python API (more information on the C API can be found in the dragon docs).

The Dragon python API uses python's `multiprocessing` API.  Dragon can therefore be used to extend scripts written for single shared memory devices with `multiprocessing` to run on multiple nodes without shared memory.

## Get Interactive Nodes

We'll run these tests in an interactive session, however a sample submit script is provided to test running in a batch job.

```shell
qsub -V -I -A alcf_training -l select=2 -l walltime=0:30:0 -l filesystems=home:eagle -q alcf_training
```

## Install

For the workshop, the prestaged environment with dragon can be loaded here:
```shell
module unload xalt
source /grand/alcf_training/workflows/_env/bin/activate
```

To create your own environment:
```shell
module unload xalt
module load conda
conda activate base
python -m venv _env
source _env/bin/activate
pip install dragonhpc
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64
```

The last installation step running `dragon-config` configures dragon to use fast RDMA transfers across Polaris' slingshot network.  Without this step, dragon would run in the default mode that uses slower TCP transfers.


## Dragon Pool

Dragon has a python API which we will use here.  The Dragon python API uses python's `multiprocessing` API.  Dragon can therefore be used to extend scripts written for single shared memory devices with `multiprocessing` to run on multiple nodes without shared memory.

This example shows how to run tasks with `Pool` in two different ways.  The first way uses `multiprocessing Pool` with the start method set to `dragon`.  This allows for pool processes to be distributed across nodes, but it is not possible to bind them to paricular gpus or cpus.

The second approach uses `dragon`'s native Pool that allows a list of dragon `Policy`s to be passed to the `Pool` that specify the gpus the processes are bound to.

These tests run a python function that sleeps and reports the hostname and GPU tile it sees pinned by `ZE_AFFINITY_MASK`.

Dragon scripts are launched with the `dragon` application, included in the demo environment.  To run the example script:

```shell
dragon 0_dragon_pool.py
```

**0_dragon_pool.py**
```python

```

## Dragon ProcessGroup

Dragon provides another way of distributing tasks to groups of workers called the dragon `ProcessGroup`.  Every process in the `ProcessGroup` can be given a specific node, gpu, and cpu, allowing for finegrained control of processes.

There are two example cases using `ProcessGroup`.  `1_dragon_process_group.py` distributes a simple python function to independent processes across the allocation.  This is done by assigning a `Policy` to each process in the group that specifies the node, gpu and cpu where it is to be run.

`2_dragon_mpi_process_group.py` uses `ProcessGroup` to run an MPI-enabled executable, where every process in the group is an MPI rank.  To enable message passing between processes in a process group, the `pmi` flag needs to be set in the `ProcessGroup` as shown in `2_dragon_mpi_process_group.py`.

## Dragon Distributed Dictionary

In addition to task launching, `dragon` provides a distributed data layer in its runtime called the `DDict` or Dragon Dictionary.  The `DDict` provisions pools of memory on nodes across the runtime where key-value data pairs can be stored.  Any process in the runtime can access any key-value pair in the `DDict`.  This is enabeld by dictionary manager processes that are created in the background within the runtime and manage the transfer of data between nodes.

`DDict`s can be used with `Pool` or `ProcessGroup`.  Multiple processes or `Pool`s and `ProcessGroup`s can use the same `DDict`.  This enables efficient sharing of data between processes in the runtime.

`3_dragon_dictionary.py` gives a simple example of how to create a `DDict` and use a `Pool` to store data within it.  The main process (running on the head node) then retrieves all the data stored in the `DDict`.

## Submitting to PBS

To submit these tests to multiple nodes with PBS, use the following submit script:

```shell
qsub 4_submit_dragon.sh
```

**3_submit_multinode.sh**
```bash
#!/bin/bash -l
#PBS -A alcf_training
#PBS -l select=2
#PBS -N dragon_test
#PBS -l walltime=0:10:00
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -l place=scatter
#PBS -q alcf_training
#PBS -V

cd $PBS_O_WORKDIR

module unload xalt
source /grand/alcf_training/workflows/_env/bin/activate

dragon 0_dragon_pool.py
sleep 1
dragon 1_dragon_process_group.py
sleep 1
dragon 2_dragon_mpi_process_group.py
sleep 1
dragon 3_dragon_dictionary.py
```