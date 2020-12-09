# Horovod Timeline Analysis for Data Parallel Training

Contact: Huihuo Zheng <huihuo.zheng@anl.gov>

This example shows the Horovod Timeline analysis. It will show the the timeline of communication in distributed training. 

To perform Horovod timeline analysis, one has to set the environment variable ```HOROVOD_TIMELINE``` which specifies the file for the output. 
  ```
  export HOROVOD_TIMELINE=timeline.json
  ```
  This file is only recorded on rank 0, but it contains information about activity of all workers. You can then open the timeline file using the `chrome://tracing` facility of the Chrome browser.

More details: https://horovod.readthedocs.io/en/stable/timeline_include.html

Visualize the results using Chrome chrome://tracing/

* Results on Theta
   - NEGOTIATION ALLREDUCE is taking a lot of time
   - ALLREDUCE is using MPI_ALLREDUCE
   - Different tensors are reduced at different time. There is an overlap between MPI and compute
   
![ThetaHorovodTimeline](ThetaHorovodTimeline.png)


* Results on ThetaGPU
   - NEGOTIATION ALLREDUCE is taking a lot of time
   - ALLREDUCE is using NCCL_ALLREDUCE
   - Different tensors are reduced at different time. There is an overlap between MPI and compute
   
![ThetaGPUHorovodTimeline](ThetaGPUHorovodTimeline.png)