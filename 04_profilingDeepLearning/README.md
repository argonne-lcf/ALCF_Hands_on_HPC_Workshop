# Profiling Deep Learning frameworks

## Communication profiling (by Huihuo Zheng)
We introduce two profiling tools for understanding the communication in distributed deep learning. 
* MPI flat profiling using mpitrace
To turn on the profiling, one has to set the following environment variable ```LD_PRELOAD```. 
```
export LD_PRELOAD=/lus/theta-fs0/software/datascience/thetagpu/hpctw/lib/libmpitrace.so
```
Then run the application as usual. MPI profiling results will be generated after the run finishes mpi_profile.XXXX.[rank_id]. 

* Horovod timeline
To perform Horovod timeline analysis, one has to set the environment variable ```HOROVOD_TIMELINE``` which specifies the file for the output. 
```
export HOROVOD_TIMELINE=timeline.json
```
This file is only recorded on rank 0, but it contains information about activity of all workers. You can then open the timeline file using the chrome://tracing facility of the Chrome browser.

More details: https://horovod.readthedocs.io/en/stable/timeline_include.html
