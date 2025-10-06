# MPI-IO and HDF5

Every application has an I/O problem... they just might not know it yet!

## Setting up

The examples are part of other repositories, so if you didn't get any submodules when you checked out, run the following commands

   git submodule init
   git submodule update

## Aurora environment

You'll need a few modules loaded:
* frameworks : Intel's Python distribution with a bunch of OneAPI-optimized modules
* hdf5/1.14.5
* mpich :  should already be in your environment by default
* parallel-netcdf

Watch out -- if you load hdf5 then frameworks, there is a serial HDF5 module in
`frameworks` that will become the default.  Instead, load `frameworks` then
`hdf5`


The `aurora-setup-env.sh` file in this directory takes care of loading these
modules in the right order.

The project name for the workshop is `alcf_training` and we have two queues
* `alcf_training` for small jobs
* `alcf_training-l` for larger jobs

For example: to submit a job you could run

    qsub -q alcf_training -A alcf_training ./job-script.sh

## Darshan

For a lot of these examples I will show [Darshan](https://www.mcs.anl.gov/research/projects/darshan/)  results.

In order to generate nice vizualizations you will need the py-darshan module:

    module load darshan-runtime darshan-util
    module load frameworks
    python -m venv darshan-venv
    source ./darshan-venv/bin/activate
    pip install darshan

Or! just source the `aurora-setup-env.sh` file to use a pre-built version for you

## MPI-IO

The I/O chapter of the MPI standard.  The most commonly available
implementation is [ROMIO](https://wordpress.cels.anl.gov/romio/).  We will talk
about ROMIO's optimizaitons and some of Cray's vendor modifications.

## Parallel-NetCDF

Widely used in climate and weather domains.  Available in the `parallel-netcdf` module

## HDF5

The "do anything" I/O library.  Lots of features, capabilities.  Load the `hdf5` module.

## other resources
This presentation builds on past seminars:

* [I/O Sleuthing](https://github.com/radix-io/io-sleuthing) : a BSSW fellowship project containing run scripts, plotting tools, and videos.
* [ATPESC Data and I/O] (https://github.com/radix-io/hands-on) : covering not only MPI-IO and HDF5 but also Parallel-NetCDF, Darshan and Globus
