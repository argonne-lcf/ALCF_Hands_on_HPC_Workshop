This example shows how to use Intel Vtune to profile the application ```pytorch_cifar10.py``` from the ```01_distributedDeepLearning``` section. Here we use Intel VTune to profile an application running on Intel CPUs (Theta).

VTune Amplifier is a full system profiler to obtain application performance details at various levels of granularity. It can be used to extract hardware performance metrics related to microarchitecute, memory, IO besides hotspots in the code. 

VTune Amplifier’s Application Performance Snapshot (APS):

VTune APS gives a high-level overview of application performance to help identify primary optimization areas. It is easy to use and provides data in HTML report.
To use,
```
$ module load vtune
$ export PMI_NO_FORK=1
```

Launch the job in interactive or batch mode:

```
$ aprun -N <ppn> -n <totRanks> [affinity opts] aps ./exe
```

Produce text and html reports:
```
$ aprun -report ./aps_result_ ….
```
A snapshot of the APS result is


VTune Profiler:

This can be used to perform various analyses on the application. For example, to collect the hotspots in the code, the following command is used
```
aprun -n 1 -N 1 amplxe-cl -c hotspots -r res_dir -- python mycode.py myarguments
```


On Theta, it is recommended to run the command to profile the code on the compute node with ```aprun```. The finalization phase is performed on the login node and the reports can be viewed there or copied back to the local machine with Intel VTune Amplifier tool installed. 


