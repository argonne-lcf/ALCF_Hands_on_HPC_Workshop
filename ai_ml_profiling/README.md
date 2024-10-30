# Profiling a simple PyTorch example
In the hands on session we will try to profile a simple PyTorch example. This mini-application (mini-benchmark) mimics the sequence parallelism strategy that we use for our auroraGPT training. Very similar strategy is adopted by Megatron-DeepSpeed as well. 

A request to an user would be to try and run the run the scripts both from an interactive session and a batch submission *before* jumping on to using the profilers! 

## Content of the repository 
In this repository we have a few scripts. We will quickly give a brief introduction to them.
- `sequence_parallelism_compute.py` \
    This will be our main workhorse for this session. In this script we have implemented a sequence parallel workflow using PyTorch tensors. This implementation interleaves collective communication with compute -- with fixed tensor parallel degree based on the total number of `MPI` ranks employed for the workload. We have implemented timing loops to measure the total execution time for each type of compute and communication. The workflow pattern that we have implemented here is:
    `ALLGATHER` --> `Matrix Multiplication 1` --> `Matrix Multiplication 2` --> `REDUCE_SCATTER`. In the docstring of the script, we see a hard coded "4". This comes from the assumption that, we have parallelized our matrix multiplication over 4 GPUs on a Polaris node.

- `qsub_polaris_sequence_parallelism.sh` \
    This is our job submission scripts. There are a few things that an user needs to modify.
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
    The `PROF_DIR` path is unused in this piece of code, as we have a separate script for launching the profiler.

    Keeping track of the stdout (.OU) and the stderror (.ER) files are important, as we plan to ask the Nsight Systems profiler to print out a summary for us. This summary comes in the .OU file.
- `nsys_wrapper.sh` \
    This is the wrapper that we will be using to run the profiler on more than 1 Node. In principle, this should be usable for deploying the profiler in more than 1 rank. An user may have to perform a `chmod +rwx` on the scripts. This script has a few directory paths, which needs to be changed accordingly. This script is set to track the Rank 0 on each node. For example, if we deploy the application in 2 Nodes, 4 Ranks (GPUs) each, then the wrapper, as is, will trace, Rank 0 and Rank 4.
- `ncu_wrapper.sh` \
    Similar to `nsys_wrapper.sh`. But for profiling a GPU kernel.
- `qsub_profile_polaris_sequence_parallelism.sh` \
    This script gives a demonstration of launching the Nsys profiler W/O the wrapper. Traces all the ranks that the application is deployed to.
- `qsub_nsys_wrapper_polaris_sequence_parallelism.sh` \
    Demonstration of how to launch the nsys profiler with a wrapper which controls the number of ranks to be traced.
- `qsub_ncu_wrapper_polaris_sequence_parallelism.sh` \
    Same as above, but for Nsight Compute (ncu) profiler.
- `sequence_parallelism_compute_lineprofile.py` \
    This script demonstrates the usage of the cuda profiler API in torch. This gives an user the finer control to profile specific chunks of code. The use of the term `lineprofile` is technically not the most appropriate, but the spirit is very similar. Intentionally left incomplete.
- `qsub_line_profile_polaris_sequence_parallelism.sh` \
    A submission script to accompany the "line profiling" like activity.

## Basic Strategy
The main code allows us to play around with a few parameters, like the number of elements in the GPU buffer (the sequence length, the number that goes in the sequence dimension of the input to a transformer layer stack), the hidden dimension (a proxy for the model dimension), and the precision type (only `float32` and `bfloat16` is supported). We will try to record results for however many combinations we can go through. The lower cutoff of expectation here is to at least get profiles for 1 and 2 nodes, for both precision type. We will also try to pay attention to the results coming out of the timing loops.

This will primarily be a two step process. Generate the profile on Polaris, `scp` or `rsync` it to the local machine (where we have `nsys` and `ncu` client/application/gui/viewer installed). Open the profiles and see if these make sense with respect to the workload that we have deployed.

Installing Nsight Systems and Nsight Compute is fairly straightforward. We can start by doing a "download Nsight Systems" google search. In any case, the two links at the bottom should take us to the desired location!

## Install Nsys

[Getting Started, Download Nsys](https://developer.nvidia.com/nsight-systems/get-started)

[Download Nsight Compute](https://developer.nvidia.com/tools-overview/nsight-compute/get-started)