# DAOS - Distributed Asynchronous Object Store File System

**Learning Goals:**

* How to use the DAOS file system on Aurora


## Overview 

[DAOS](https://docs.alcf.anl.gov/aurora/data-management/daos/daos-overview/) is a major file system composed of 1024 DAOS server storage nodes fully integrated with the wider Aurora compute fabric. 
Compared to the Lustre file system `flare`, DAOS will have twice the storage capacity (230 PB vs 100 PB) and more than an order of magnitude higher theoretical peak bandwidth (30 TB/s vs 0.6 TB/s).

While we are working towards bringing up the entire 1024 DAOS server available users, our current `daos_user` cluster has 128 / 1024 DAOS nodes and can deliver up to 5 TB/s. 
These are the expected theoretical peak bandwidths for DAOS clusters of different sizes: 

DAOS Nodes  | Percentage of Total DAOS Nodes | Theoretical Peak Bandwidth
| -- | -- |  --
20   |  2%    | 1 TB/s
128  | 12.50% | 5 TB/s
600  | 60%    | 10 TB/s
800  | 78%    | 20 TB/s
1024 | 100%   | 30 TB/s



DAOS storage is organized into POOLS and CONTAINERS:

- a DAOS **POOL** is a physically allocated dedicated storage space for a project. Pools are managed by administrators. 
- a DAOS **CONTAINER** is a collection of data objects of different types: for example, POSIX Containers include files and directories. Containers can be created and managed by users inside a DAOS pool: typically a user should not need more than one or a few containers (each container can contain millions of folders and files).

Here we show how you can use DAOS for fast I/O and checkpointing on Aurora.
These are the steps we will take:

1. Load the DAOS module
1. Request a DAOS pool space allocated for your project
1. Create a POSIX container 
1. Mount a POSIX container on a login node and on compute nodes
1. Submit a job with access to the DAOS file system

> **Note**: This is an initial test DAOS configuration. As such, data may be removed or unavailable at any time, so make sure to back up all important data to `flare`.



## DAOS Module

To use DAOS you should load the `daos` module. 

```bash
module use /soft/modulefiles
module load daos/base
```
This can be done on a login node (UAN) or on a compute node:

> **Note:** To ensure that DAOS is accessible from the compute nodes, use the following arguments in your `qsub` job submission:
>  - `-l filesystems=daos_user`
>  - `-l daos=daos_user`

To see a list of available DAOS pools:
```bash
$ daos pool list
Pool            Size   State    Used Imbalance Disabled 
----            ----   -----    ---- --------- -------- 
datascience     96 TB  Degraded 11%  11%       32/4096  
```

If you do not have any pool, you can request a pool allocation.


## Request a DAOS Pool Allocation

The first step in using DAOS is to get a DAOS pool, that is a physically allocated dedicated storage space for your project. 

If your project does not have a DAOS pool yet, you can email [support@alcf.anl.gov](mailto:support@alcf.anl.gov) to submit a request with the following information:

* Project Name
* ALCF User Names
* Total Space requested (typically 100s TB)
* Justification
* Preferred pool name

> **Note**: the storage space of a DAOS pool cannot be extended, so make sure to request enough size for your project. 

Once a pool is allocated for your project space, confirm you are able to query the pool via:
```bash
daos pool query <pool_name>
```

Example:
```bash
$ daos pool query datascience
Pool 751f934f-9ab3-4611-b10e-4d2fc9ac0129, ntarget=4096, disabled=32, leader=190, version=67, state=Degraded
Pool health info:
- Rebuild done, 0 objs, 0 recs
Pool space info:
- Target(VOS) count:4064
- Storage tier 0 (SCM):
  Total size: 3.0 TB
  Free: 2.6 TB, min:644 MB, max:649 MB, mean:649 MB
- Storage tier 1 (NVMe):
  Total size: 96 TB
  Free: 96 TB, min:21 GB, max:24 GB, mean:24 GB
```

To check the status of DAOS, use the [DAOS sanity check](https://docs.alcf.anl.gov/aurora/data-management/daos/daos-overview/#daos-sanity-checks) commands.


## DAOS Container

The container is the basic unit of storage. A POSIX container can contain hundreds of millions of files, you can use it to store all of your data. You only need a small set of containers; usually just one per major unit of project work is sufficient.

There are [3 modes](https://docs.alcf.anl.gov/aurora/data-management/daos/daos-overview/#daos-container) with which we can operate with the DAOS containers, in this tutorial we will cover only the POSIX mode.

To list all containers available in a pool:
```bash
daos cont list <pool_name>
```

Example
```bash
$ daos cont list datascience
UUID                                 Label      
----                                 -----      
64125a7c-4750-4ef1-bd02-af04c8b246a8 LLM-GPT-1T 
22677585-b8f1-4616-b73b-375f91674a94 ior_1      
d106b9f3-2f78-4699-8f10-dc285d134da5 softwares
```

### Create a POSIX container

You can create a container from a login node (recommended) or a compute node. 
```bash
DAOS_POOL=datascience
DAOS_CONT=test_container
daos container create --type POSIX ${DAOS_POOL} ${DAOS_CONT} --properties rd_fac:1
```
Example output:
```bash
Successfully created container 6e0f0253-8cb3-494b-81df-439e7ba89b75
  Container UUID : 6e0f0253-8cb3-494b-81df-439e7ba89b75
  Container Label: test_container                      
  Container Type : POSIX     
```

- If you prefer no data protection and recovery, you can remove `--properties rd_fac:1`. We recommend to have at least `--properties rd_fac:1`.
- To destroy a container use
  ```bash
  daos cont destroy ${DAOS_POOL} ${DAOS_CONT}
  ```

 
### Mount a POSIX container on a login node

```bash
mkdir –p /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}
# To mount
start-dfuse.sh -m /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT} --pool ${DAOS_POOL} --cont ${DAOS_CONT}
mount | grep dfuse # To confirm if its mounted

# List the content of the container
ls /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}

# Copy a file to the container
echo "hello" > temp.txt
cp temp.txt /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}/
cat /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}/temp.txt

# To unmount
fusermount3 -u /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT} 
```

### Mount a POSIX container on all compute nodes

You need to mount the container on all compute nodes.

```bash
# Use pdsh to mount the container at /tmp/<pool>/<container> on all compute nodes
launch-dfuse.sh ${DAOS_POOL}:${DAOS_CONT}

# To confirm it is mounted
mount | grep dfuse  
ls /tmp/${DAOS_POOL}/${DAOS_CONT}/

# To unmount on all nodes
clean-dfuse.sh ${DAOS_POOL}:${DAOS_CONT}  
```


## Job Submission

Example of an interactive job with DAOS on 2 nodes, with arguments `-l filesystems=daos_user -l daos=daos_user` to ensure that DAOS is accessible from the compute nodes:
```bash
qsub -l filesystems=home:flare:daos_user -l daos=daos_user -l select=2 -l walltime=00:30:00 -A alcf_training -k doe -q aurorabootcamp -I
```

Use the following arguments in your `mpiexec` command:

- Currently, `--no-vni` is required in the `mpiexec` command to use DAOS.
- To achieve good performance, set the [NIC and Core Binding](https://docs.alcf.anl.gov/aurora/data-management/daos/daos-overview/#nic-and-core-binding). For `-ppn 12`, the following binding is recommended: `--cpu-bind=list:4:9:14:19:20:25:56:61:66:71:74:79`.

Check out the script [./examples/daos/daos_torch.sh](examples/daos/daos_torch.sh) for a simple example on how to use `mpiexec` to save PyTorch tensors to a DAOS container.


## Additional Resources

- [ALCF Training Video: *Overview of DAOS and Best Practices*](https://www.alcf.anl.gov/support-center/training/overview-daos-and-best-practices)
- [Official DAOS documentation](https://docs.daos.io/v2.6/overview/architecture/)
- [Additional information on moving data from/to DAOS](https://docs.alcf.anl.gov/aurora/data-management/moving_data_to_aurora/daos_datamover/).

## [NEXT -> JAX](jax.md)
