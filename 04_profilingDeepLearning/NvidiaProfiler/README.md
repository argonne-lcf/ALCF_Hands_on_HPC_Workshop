
Example to profile DL codes on ThetaGPU using Nvidia profiler tools, namely, NSight and DLProf.

Nsight tools can be used to profile and analyze a wide-variety of applications, while DLProf is aimed for deep learning applications.

(1) NSight tools

Nsight Systems - System-wide application algorithm tuning \
Nsight Compute - Debug CUDA API and optimize CUDA kernels

The script ```submit_thetagpu.sh``` provides instructions to profile ```tensorflow2_cifar10.py``` code from ```01_distributedDeepLearning``` section.

To profile with Nsight systems (https://developer.nvidia.com/nsight-systems) (refer to Step A in the ```submit_thetagpu.sh``` script). A typical command to profile an application, say train.py, is
```
$ nsys profile python myapp.py
```
This generates profile file in .qdrep format which can be imported to view with Nsight Systems UI. It is recommended to run the script to generate the profile file
on the compute node and copy it to local machine with Nvidia Nsight tools installed to import the profile and view the analysis.


To view the profiling results in GUI, follow the commands in the section "Steps to visualize results (Step C)" in the script.

The profile for ```tensorflow2_cifar10.py``` looks like this. It shows the kernels generated with more details on a timeline trace.

![Alt text](./nsys-trace.png?raw=true)


Nsight Compute (https://developer.nvidia.com/nsight-compute) is an interactive kernel profiler for CUDA applications. It provides detailed performance metrics and API debugging via a user interface and command line tool.

The command used to profile using Nsight Compute is
```
ncu python myapp.py
```

Usually, this method incurs lot of overhead in collecting the performance metrics. To help minimize, we opt for selective profiling where we profile only selected kernels. For example, if we want to profile kernels that perform gemm operations, we use

```
ncu --kernel-id ::regex:gemm: python myapp.py
```

For the code ```tensorflow2_cifar10.py```, the profile from Nsight Compute yields the following metrics.

![Alt text](./Nsight-compute.png?raw=true)


Try varying batch size with --batch_size parameter and observe the difference in metrics. For batch sizes 4 and 512, these are

![Alt text](./ncu-comparison.jpg?raw=true)


(2) DLProf

Deep Learning Profiler (DLProf) (https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/index.html), is a tool for profiling deep learning models to analyze and improve performance of their models visually via the DLProf Viewer or by analyzing text reports.

Few important features:
- Measure if an operation can run on Tensorcores and measure their usage if kernels are running on them.
- Use NVTX Markers to profile only a portion of code,such  as dataloading stage or the training phase.
- Analyze XLA compiled TensorFlow models.

To use ```dlprof``` on the example code we have, use the script with dlprof argument ```submit_thetagpu.sh dlprof```

This will profile the model and print out the summary in results folder in a bunch of csv files. These include details such as summary, kernel, detailed_op, tensor_core, op_type, etc. 

For example, the ```dlprof_tensor_core.csv``` file will list out kernels that are running on tensor cores and their utilization.


Another tool ```dlprofviewer``` can also be used to visualize the results in GUI. Refer to https://docs.nvidia.com/deeplearning/frameworks/tensorboard-plugin-user-guide/index.html for more details.
