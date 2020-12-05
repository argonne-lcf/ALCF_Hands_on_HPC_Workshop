Example to profile DL codes on ThetaGPU using Nvidia profiler tools

Nsight Systems - System-wide application algorithm tuning \
Nsight Compute – Debug CUDA API and optimize CUDA kernels

To profile with Nsight systems (refer to Step A in the ```submit_thetagpu.sh``` script). A typical command is
```
$ nsys profile python train.py
```
This generates profile file in ‘report.qdrep’ which can be imported to view with Nsight Systems UI. It is recommended to run the script to generate the profile file
on the compute node and copy it to local machine with Nvidia Nsight tools installed to import the profile and view the analysis.

To profile with Nsight Compute (refer to Step B in the ```submit_thetagpu.sh``` script). This is invoked by
```
$ ncu profile python train.py
```

To view the profiling results in GUI, follow the commands in the section "Steps to visualize results (Step C)" in the script.


