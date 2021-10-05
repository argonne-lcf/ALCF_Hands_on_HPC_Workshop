# MPI profiling for Data Parallel Training

Contact: Huihuo Zheng <huihuo.zheng@anl.gov>

In the distributed training, all the workers compute the loss and gradients with respect to the sub-minibatch of dataset. It then synchronize the loss and gradients across all the workers. This is typicall done by communication libraries such as MPI or NCCL, CCL, Gloo. 

In this example, we would like to do a flat profiling on the MPI communication involved in the training. We use a library called [mpitrace]( https://github.com/IBM/mpitrace) by Bob Walker (IBM). The only thing you need to do is setting the LD_PRELOAD

* On ThetaGPU
```bash
export LD_PRELOAD=/lus/theta-fs0/software/datascience/thetagpu/hpctw/lib/libmpitrace.so
```
* On Theta
```bash
export LD_PRELOAD=/soft/perftools/hpctw/NONE/libhpmprof.so
```
Then after the run, in your job directory, it will output the MPI profiling results mpi_profile.XXXXX.RANKID. It shows a histogram of the MPI communications. 


To run the examples
* Theta
```bash
cd theta/
qsub qsub.sc
```
* ThetaGPU
```bash
cd thetagpu/
qsub qsub_cpu.sc
qsub qsub_gpu.sc
```

The following show the results for [pytorch_cifar10.py](./pytorch_cifar10.py) example we provided in the distributed data parallel training. We find that 
* On CPU, the MPI communication involves are mostly MPI_Allreduce. In this example, 8 byte Allreduce dominate the communication. This is from negotiation stage where the application is checking the tensors accross the workers which are ready to be reduced. 
* On GPU, however, we do not see too much MPI_Allreduce, since the communication is done through NCCL. 

## Profiling results
### Theta CPU (5 epochs)
```bash
Data for MPI rank 0 of 32:
Times and statistics from MPI_Init() to MPI_Finalize().
-----------------------------------------------------------------------
MPI Routine                        #calls     avg. bytes      time(sec)
-----------------------------------------------------------------------
MPI_Comm_rank                           5            0.0          0.000
MPI_Comm_size                           3            0.0          0.000
MPI_Bcast                             261       713382.3          0.743
MPI_Allreduce                       47108       247017.5        250.492
MPI_Gather                            112            4.0          0.085
MPI_Gatherv                           112            0.0          0.007
MPI_Allgather                           2            4.0          0.000
-----------------------------------------------------------------
MPI task 0 of 32 had the minimum communication time.
total communication time = 251.328 seconds.
total elapsed time       = 317.922 seconds.
user cpu time            = 8761.318 seconds.
system time              = 308.435 seconds.
max resident set size    = 1177.191 MBytes.

Rank 3 reported the largest memory utilization : 1177.88 MBytes
Rank 0 reported the largest elapsed time : 317.92 sec

-----------------------------------------------------------------
Message size distributions:

MPI_Bcast                 #calls    avg. bytes      time(sec)
                             117           4.0          0.005
                             101          25.0          0.002
                               2          40.0          0.000
                               2         256.0          0.000
                               6         283.7          0.000
                               8         867.2          0.000
                               3        1573.7          0.000
                               1        2121.0          0.000
                               3        6232.3          0.000
                               4       16384.0          0.046
                               2      163840.0          0.002
                               2      442368.0          0.051
                               6     2850816.0          0.085
                               2    16777216.0          0.114
                               2    67108864.0          0.437

MPI_Allreduce             #calls    avg. bytes      time(sec)
                              20           4.0          0.001
                           45773           8.0        214.933
                             124          40.0          0.137
                              16         256.0          0.002
                              43         952.6          0.006
                              12        1536.0          0.002
                             124        7135.0          0.026
                             125       16384.0          0.282
                             125      163840.3          0.954
                             124      443061.7          0.230
                             372     2851875.8          3.562
                             125    16865650.7          5.794
                             125    67108864.0         24.563

MPI_Gather                #calls    avg. bytes      time(sec)
                             112           4.0          0.085

MPI_Allgather             #calls    avg. bytes      time(sec)
                               2           4.0          0.000

-----------------------------------------------------------------

Communication summary for all tasks:

  minimum communication time = 251.328 sec for task 0
  median  communication time = 273.408 sec for task 26
  maximum communication time = 278.406 sec for task 28


MPI timing summary for all ranks:
taskid       hostname     comm(s)  elapsed(s)     user(s)   system(s)    size(MB)    switches
     0       nid00102     251.33      317.92     8761.32      308.44     1177.19      634913
     1       nid00102     267.30      317.92     4895.77      158.60     1168.41      257482
     2       nid00102     269.22      317.92     4937.79      160.15     1165.58      253738
     3       nid00102     269.65      317.92     5059.87      157.44     1177.88      265483

```


### ThetaGPU - GPU

```bash
Data for MPI rank 0 of 8:
Times and statistics from MPI_Init() to MPI_Finalize().
-----------------------------------------------------------------------
MPI Routine                        #calls     avg. bytes      time(sec)
-----------------------------------------------------------------------
MPI_Comm_rank                           5            0.0          0.000
MPI_Comm_size                           3            0.0          0.000
MPI_Bcast                              64          129.3          0.000
MPI_Barrier                             1            0.0          0.001
MPI_Allreduce                        8758            8.0          6.038
MPI_Gather                             29            4.0          0.007
MPI_Gatherv                            29            0.0          0.000
MPI_Allgather                           2            4.0          0.000
-----------------------------------------------------------------
total communication time = 6.047 seconds.
total elapsed time       = 49.737 seconds.
user cpu time            = 28.839 seconds.
system time              = 58.022 seconds.
max resident set size    = 4025.664 MBytes.

Rank 2 reported the largest memory utilization : 4045.00 MBytes
Rank 1 reported the largest elapsed time : 49.85 sec

-----------------------------------------------------------------
Message size distributions:

MPI_Bcast                 #calls    avg. bytes      time(sec)
                              34           4.0          0.000
                              17          25.0          0.000
                               1         128.0          0.000
                               8         211.0          0.000
                               1         281.0          0.000
                               1         569.0          0.000
                               2        2525.0          0.000

MPI_Allreduce             #calls    avg. bytes      time(sec)
                              40           4.0          0.002
                            8718           8.0          6.036
```

### ThetaGPU - CPU

```bash
Data for MPI rank 0 of 8:
Times and statistics from MPI_Init() to MPI_Finalize().
-----------------------------------------------------------------------
MPI Routine                        #calls     avg. bytes      time(sec)
-----------------------------------------------------------------------
MPI_Comm_rank                           5            0.0          0.000
MPI_Comm_size                           3            0.0          0.000
MPI_Bcast                              91      2046004.3          0.136
MPI_Allreduce                      123445       739019.0        109.085
MPI_Gather                             27            4.0          0.001
MPI_Gatherv                            27            0.0          0.000
MPI_Allgather                           2            4.0          0.000
-----------------------------------------------------------------
MPI task 0 of 8 had the median communication time.
total communication time = 109.223 seconds.
total elapsed time       = 671.970 seconds.
user cpu time            = 883.652 seconds.
system time              = 583.520 seconds.
max resident set size    = 820.883 MBytes.

Rank 2 reported the largest memory utilization : 839.72 MBytes
Rank 2 reported the largest elapsed time : 671.97 sec

-----------------------------------------------------------------
-----------------------------------------------------------------
Message size distributions:

MPI_Bcast                 #calls    avg. bytes      time(sec)
                              32           4.0          0.000
                              12          25.0          0.000
                               2          40.0          0.000
                              15         219.5          0.000
                               6         938.7          0.000
                               2        1536.0          0.000
                               2        2525.0          0.000
                               2        6912.0          0.000
                               4       16384.0          0.002
                               2      163840.0          0.001
                               2      442368.0          0.001
                               6     2850816.0          0.024
                               2    16777216.0          0.023
                               2    67108864.0          0.084

MPI_Allreduce             #calls    avg. bytes      time(sec)
                              40           4.0          0.000
                          115190           8.0         12.123
                              24          40.0          0.000
                               6         256.0          0.000
                              18        1024.0          0.000
                               9        1536.0          0.000
                             648        7181.0          0.030
                             980       16384.0          0.081
                             980      163879.0          0.319
                             980      445554.2          0.640
                            2610     2914170.0          9.014
                             980    17588391.2         17.817
                             980    67108864.0         69.061
```