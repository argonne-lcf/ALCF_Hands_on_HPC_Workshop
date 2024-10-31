# Profiling a simple PyTorch example
In the hands on session we will try to profile a simple PyTorch example. This mini-application (or "mini-benchmark") mimics the sequence parallelism strategy that we use for our AuroraGPT training. A similar strategy is adopted 
by [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) as well. 

Users should first attempt to run the following workload scripts without profiling on a Polaris compute node (from either an interactive session or a batch job submission). 

## Content of the repository 
In this repository we have a few scripts. We will quickly give a brief introduction to them.
- `sequence_parallelism_compute.py` \
    This will be our main workhorse for this session. In this script we have implemented a sequence parallel workflow using PyTorch tensors. This implementation interleaves collective communication with compute with fixed tensor parallel degree based on the total number of `MPI` ranks employed for the workload. We have implemented timing loops to measure the total execution time for each type of compute and communication. The workflow pattern that we have implemented here is:
    `ALLGATHER` --> `Matrix Multiplication 1` --> `Matrix Multiplication 2` --> `REDUCE_SCATTER`. In the docstring of the script, we see a hard coded "4". This comes from the assumption that, we have parallelized our matrix multiplication over 4 GPUs on a Polaris node.

- `qsub_polaris_sequence_parallelism.sh` \
    This is our job submission script. There are a few things that an user needs to modify.
    ```
    #PBS -l place=scatter
    #PBS -l walltime=00:05:00
    #PBS -q HandsOnHPC
    #PBS -l filesystems=home:grand
    #PBS -A alcf_training
    #PBS -N the_name_you_like
    #PBS -k doe
    #PBS -o /path/to/the/stdout
    #PBS -e /path/to/the/stderror

    ...
    WORK_DIR=/path/to/sequence_parallelism.py
    PROF_DIR=/path/to/the/directory/for/profiles
    ```
    Keeping track of the stdout (`.OU`) and the stderror (`.ER`) files are important, as we plan to ask the NVIDIA Nsight Systems profiler to print out a summary for us. This summary comes in the `.OU` file.
- `nsys_wrapper.sh` \
    This is the wrapper that we will be using to run the profiler on more than 1 node. In principle, this should be usable for deploying the profiler in more than 1 rank. An user may have to perform a `chmod +rwx` on the scripts. This script has a few directory paths, which all needs to be changed accordingly. This script is set to track the Rank 0 on each node. For example, if we deploy the application in 2 nodes, 4 ranks (GPUs) each, then the wrapper, as is, will trace, Rank 0 and Rank 4.
- `ncu_wrapper.sh` \
    Similar to `nsys_wrapper.sh`, but for profiling a GPU kernel.
- `qsub_profile_polaris_sequence_parallelism.sh` \
    This script gives a demonstration of launching the `nsys` profiler without the wrapper. It traces all the ranks that the application is deployed to.
- `qsub_nsys_wrapper_polaris_sequence_parallelism.sh` \
    Demonstration of how to launch the `nsys` profiler with a wrapper which controls the number of ranks to be traced.
- `qsub_ncu_wrapper_polaris_sequence_parallelism.sh` \
    Same as above, but for Nsight Compute (ncu) profiler.
- `sequence_parallelism_compute_lineprofile.py` \
    This script demonstrates the usage of the CUDA profiler API in torch. This gives an user the finer control to profile specific chunks of code. Intentionally left incomplete.
- `qsub_line_profile_polaris_sequence_parallelism.sh` \
    A submission script to accompany the "line profiling" like activity.

## Basic Strategy
The main code allows us to play around with a few parameters, like the number of elements in the GPU buffer (the sequence length, the number that goes in the sequence dimension of the input to a transformer layer stack), the hidden dimension (a proxy for the model dimension), and the precision type (only `float32` and `bfloat16` is supported). We will try to record results for however many combinations we can go through. The lower cutoff of expectation here is to at least get profiles for 1 and 2 nodes, for both precision types. We will also try to pay attention to the results coming out of the timing loops.

This will primarily be a two step process. Generate the profile on Polaris, `scp` or `rsync` it to the local machine (where we have `nsys` and `ncu` client/application/gui/viewer installed). Open the profiles and see if these make sense with respect to the workload that we have deployed.

Follow the links below to install a version of NVIDIA's Nsight Systems and Nsight Compute on your local machine.

## Install Nsys

[Getting Started, Download Nsys](https://developer.nvidia.com/nsight-systems/get-started)

[Download Nsight Compute](https://developer.nvidia.com/tools-overview/nsight-compute/get-started)

## Viewing the profile

After the successful completion of the execution of the job script, we should 
see the following message from each of the traced ranks:

For `nsys` profiler reports
```
# From Rank 0

 Generated:
x3007c0s13b0n0.hsn.cm.polaris.alcf.anl.gov 0: /home/hossainm/hpc_workshop_october_2024/profiles/nsys_seq_parallel_bf16_n2_2024-10-30_134546/sequence_parallelism_compute_n2_2024-10-30_134546/nsys_seq_parallel_bf16_n2_2024-10-30_134546_0.nsys-rep
x3007c0s13b0n0.hsn.cm.polaris.alcf.anl.gov 0: /home/hossainm/hpc_workshop_october_2024/profiles/nsys_seq_parallel_bf16_n2_2024-10-30_134546/sequence_parallelism_compute_n2_2024-10-30_134546/nsys_seq_parallel_bf16_n2_2024-10-30_134546_0.sqlite
```

For `ncu` profiler reports
```
# From Rank 4

x3005c0s25b1n0.hsn.cm.polaris.alcf.anl.gov 4: ==PROF== Report: /home/hossainm/hpc_workshop_october_2024/profiles/ncu_seq_parallel_bf16_n2_2024-10-30_135414/sequence_parallelism_compute_n2_2024-10-30_135414/ncu_seq_parallel_bf16_n2_2024-10-30_135414_4.ncu-rep
```

Then, we can use either the `scp` or `rsync` to get the reports to our local 
machine:

```
scp [USERNAME]@polaris.alcf.anl.gov:/path/to/the/profles/file.nsys-rep /path/to/local/machine

or 

rsync -avh --progress [USERNAME]@polaris.alcf.anl.gov:/path/to/the/profles/file.nsys-rep /path/to/local/machine 
```

The next step is to load the `nsys-rep` files in the Nsight Systems GUI, and 
the `ncu-rep` files to the Nsight Compute GUI. 

### For a single rank run

#### `nsys` profiles
In the single rank case, we go to the top left, go `file` --> `open` and select
the file that we want to look at. For this particular example, we have focused
on the GPU activities. This activity is shown on the second column from the 
left, named as `CUDA HW ...`. If we expand the `CUDA HW ...` tab, we find an
`NCCL` tab. This tab shows the communicaltion library calls. This is of 
importance for this example because of the collective communication involved.

#### `ncu` profiles
The primary qualitative distinction between the `nsys-rep` files and the 
`ncu-rep` files is that, the `nsys-rep` file presents data for the overall 
execution of the application, whereas  the `ncu-rep` file presents data for the
execution of one particular kernel. Our setup here traces only one kernel, but
multiple kernels could be traced at a time, but that can become a time consuming
process.

We use the `--stats=true --show-output=true`(see `nsys_wrapper.sh`) 
options while collecting the 
`nsys` data. As a result, we get a system-wide summary in our `.OU` files 
(if run with a job submission script, otherwise on the terminal), and find the
names of the kernels that has been called/used for compute and communication. 
Often we would start with investigating the kernels that have been called the 
most times or the ones where we spent the most time executing them. In this 
particular instance we chose to analyze the `gemm` kernels, which are related 
to the matrix multiplication. The full name of this kernel is passed to the 
`ncu` profiler with the option `-k` (see `ncu_wrapper.sh`).

Loading the `ncu-rep` files works similarly as the `nsys-rep` files. Here, the 
important tab is the `Details` tab. We find that at the 3rd row from the top.
Under that tab we have the `GPU Speed of Light Throughput` section. In this 
section we can find plots showing GPU compute and memory usage. On the right 
hand side of the tab, there is a menu bar which gives us the option to select
which plot to display, either the roofline plot or the compute-memory 
throughput chart.

### For a multi-rank run

#### `nsys` profiles
In the case, where we have traced multiple ranks, whether from a single node or
multiple nodes `nsys` GUI allow us to view the reports in a combined fashion on
a single timeline (same time-axis for both reports). This is done through the
"multi-report view", `file` --> `New multi-report view` or `file` --> `Open` 
and selecting however many reports we would like to see in a combined timeline,
`nsys` prompts the user to allow for a "multi-report view". These can also be 
viewed separately.

## Profiler Options
In both cases, `nsys` and `ncu` we have used the standard option sets to 
generate the profiles. The exhaustive list could be found in the respective
documentation pages:

[Nsight System User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
[Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

There are many other information provided through these reports. Here we have 
discussed the way to view the high level information.


## Box link with profile examples
In this Box link we have a few example profiles available for the users. The 
intent here is to use if we can not generate the profiles during the hands on 
session.

[Box link for profiles](https://anl.box.com/s/qb088ojo9dyg4lcfl6y5q3oeh6jbze4q)
