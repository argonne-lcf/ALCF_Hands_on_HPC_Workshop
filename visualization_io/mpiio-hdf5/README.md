# MPI-IO and HDF5

Every application has an I/O problem... they just might not know it yet!

## Setting up

The examples are part of other repositories, so if you didn't get any submodules when you checked out, run the following commands

   git submodule init
   git submodule update

## Polaris environment

You'll need a few modules loaded:
* cray-python
* cray-hdf5-parallel
* cray-mpich
* cray-parallel-netcdf

The project name for the workshop is `alcf_training` and we have two queues
* HandsOnHPC for single-node jobs
* HandsOnHPCScale for multi-node (up to 128 nodes) jobs

For example: to submit a job you could run

    qsub -q HandsOnHPC -A alcf_training ./job-script.sh

## MPI-IO

The I/O chapter of the MPI standard.  The most commonly available
implementation is [ROMIO](https://wordpress.cels.anl.gov/romio/).  We will talk
about ROMIO's optimizaitons and some of Cray's vendor modifications.

## Parallel-NetCDF

Widely used in climate and weather domains

## HDF5

The "do anything" I/O library.  Lots of features, capabilities.

## other resources
This presentation builds on past seminars:

* [I/O Sleuthing](https://github.com/radix-io/io-sleuthing) : a BSSW fellowship project containing run scripts, plotting tools, and videos.
* [ATPESC Data and I/O] (https://github.com/radix-io/hands-on) : covering not only MPI-IO and HDF5 but also Parallel-NetCDF, Darshan and Globus
