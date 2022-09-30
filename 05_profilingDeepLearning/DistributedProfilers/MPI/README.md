# MPI profiling for Data Parallel Training

Contact: Denis Boyda <dboyda@anl.gov>

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


To run the examples at ThetaGPU
```bash
qsub qsub_gpu.sh
```
or get a node in interactive regime
```bash
qsub -A SDL_Workshop -q training-gpu -n 1 -t 60 -I
```
and run with `gpu_num`
```bash
./run_gpu.sh gpu_num
```

The following show the results for [tensorflow2_mnist.py](../tensorflow2_mnist.py) example. We find that 
* On CPU, the MPI communication involves are mostly MPI_Allreduce. In this example, 8 byte Allreduce dominate the communication. This is from negotiation stage where the application is checking the tensors accross the workers which are ready to be reduced. 
* On GPU, however, we do not see too much MPI_Allreduce, since the communication is done through NCCL. 

## Profiling results
### ThetaGPU - CPU
```bash
Data for MPI rank 0 of 16:
Times and statistics from MPI_Init() to MPI_Finalize().
-----------------------------------------------------------------------
MPI Routine                        #calls     avg. bytes      time(sec)
-----------------------------------------------------------------------
MPI_Comm_rank                           5            0.0          0.000
MPI_Comm_size                           3            0.0          0.000
MPI_Bcast                             123       117124.9          0.018
MPI_Allreduce                        8161       138800.2         10.575
MPI_Gather                             49            4.0          0.006
MPI_Gatherv                            49            0.0          0.002
MPI_Allgather                           2            4.0          0.000
-----------------------------------------------------------------
MPI task 0 of 16 had the minimum communication time.
total communication time = 10.602 seconds.
total elapsed time       = 15.548 seconds.
user cpu time            = 131.483 seconds.
system time              = 6.319 seconds.
max resident set size    = 1655.000 MBytes.

Rank 3 reported the largest memory utilization : 1873.82 MBytes
Rank 7 reported the largest elapsed time : 15.55 sec

-----------------------------------------------------------------
Message size distributions:

MPI_Bcast                 #calls    avg. bytes      time(sec)
                              49           4.0          0.000
                               1           8.0          0.000
                              45          25.0          0.000
                               3          40.0          0.000
                               3         128.0          0.000
                               3         256.0          0.000
                               4         470.2          0.000
                               1         705.0          0.000
                               3        1152.0          0.000
                               2        2697.0          0.000
                               3        5120.0          0.000
                               3       73728.0          0.001
                               3     4718592.0          0.017

MPI_Allreduce             #calls    avg. bytes      time(sec)
                            7428           8.0          5.696
                              11          40.0          0.003
                               1         256.0          0.000
                               4         512.0          0.003
                               9         552.0          0.004
                              25        1536.0          0.006
                             211        5642.8          0.255
                             236       75131.7          0.209
                             236     4719155.9          4.400

MPI_Gather                #calls    avg. bytes      time(sec)
                              49           4.0          0.006

MPI_Allgather             #calls    avg. bytes      time(sec)
                               2           4.0          0.000

-----------------------------------------------------------------

Communication summary for all tasks:

  minimum communication time = 10.602 sec for task 0
  median  communication time = 11.834 sec for task 5
  maximum communication time = 12.177 sec for task 3


MPI timing summary for all ranks:
taskid       hostname     comm(s)  elapsed(s)     user(s)   system(s)    size(MB)    switches
     0     thetagpu24      10.60       15.55      131.48        6.32     1655.00      171601
     1     thetagpu24      11.69       15.55      128.59        7.10     1601.46      171001
     2     thetagpu24      10.65       15.55      128.62        6.48     1690.21      172943
     3     thetagpu24      12.18       15.55      130.75        6.56     1873.82      170879
     4     thetagpu24      11.84       15.55      129.73        7.06     1807.09      170934
     5     thetagpu24      11.83       15.55      127.51        7.38     1768.41      171018
     6     thetagpu24      11.86       15.55      128.02        7.64     1768.84      173285
     7     thetagpu24      11.91       15.55      131.22        6.95     1644.43      171220
     8     thetagpu24      11.90       15.55      131.91        6.27     1785.00      172127
     9     thetagpu24      11.67       15.55      128.28        6.99     1589.02      170993
    10     thetagpu24      11.53       15.55      128.25        6.94     1650.37      170456
    11     thetagpu24      11.83       15.55      129.37        6.73     1621.89      169177
    12     thetagpu24      11.49       15.55      127.99        7.49     1732.30      171916
    13     thetagpu24      12.11       15.55      128.51        8.25     1538.35      169615
    14     thetagpu24      12.08       15.55      128.38        7.36     1738.74      169231
    15     thetagpu24      12.11       15.55      130.50        7.19     1869.61      172182

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
MPI_Bcast                              21          272.9          0.001
MPI_Barrier                             1            0.0          0.077
MPI_Allreduce                       11658            8.0          5.440
MPI_Gather                             10            4.0          0.000
MPI_Gatherv                            10            0.0          0.000
MPI_Allgather                           2            4.0          0.000
-----------------------------------------------------------------
total communication time = 5.519 seconds.
total elapsed time       = 17.725 seconds.
user cpu time            = 14.675 seconds.
system time              = 2.892 seconds.
max resident set size    = 4230.734 MBytes.

Rank 0 reported the largest memory utilization : 4230.73 MBytes
Rank 3 reported the largest elapsed time : 17.84 sec

-----------------------------------------------------------------
Message size distributions:

MPI_Bcast                 #calls    avg. bytes      time(sec)
                              10           4.0          0.000
                               7          25.0          0.000
                               1         128.0          0.001
                               2         861.0          0.000
                               1        3665.0          0.000

MPI_Allreduce             #calls    avg. bytes      time(sec)
                           11658           8.0          5.440

MPI_Gather                #calls    avg. bytes      time(sec)
                              10           4.0          0.000

MPI_Allgather             #calls    avg. bytes      time(sec)
                               2           4.0          0.000

-----------------------------------------------------------------

Communication summary for all tasks:

  minimum communication time = 3.601 sec for task 2
  median  communication time = 5.462 sec for task 6
  maximum communication time = 8.355 sec for task 7


MPI timing summary for all ranks:
taskid       hostname     comm(s)  elapsed(s)     user(s)   system(s)    size(MB)    switches
     0     thetagpu19       5.52       17.73       14.67        2.89     4230.73       61734
     1     thetagpu19       7.00       17.84       16.14        3.02     4217.52       68565
     2     thetagpu19       3.60       17.84       13.09        2.79     4218.41       72360
     3     thetagpu19       5.15       17.84       14.47        2.82     4210.53       69541
     4     thetagpu19       4.44       17.84       13.70        2.88     4190.15       65634
     5     thetagpu19       6.15       17.84       15.52        2.83     4205.98       60704
     6     thetagpu19       5.46       17.84       14.84        2.76     4210.84       62171
     7     thetagpu19       8.35       17.73       17.54        3.01     4189.67       66328
```
