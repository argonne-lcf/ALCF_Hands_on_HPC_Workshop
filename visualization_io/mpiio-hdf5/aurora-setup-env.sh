# modules needed for ATPESC 2025 I/O examples on aurora.alcf.anl.gov
module add parallel-netcdf
module add frameworks
module add hdf5
# we are using a custom Darshan install for this trainig; the darshan module
# availale in the default module path as of July 25, 2025 has an unusual
# configuration and does not auto-instrument applications
module use /lus/flare/projects/alcf_training/io-libraries/modulefiles/
module load darshan-runtime/3.5.0

# point to manual PyDarshan install in ATPESC scratch space
export PYTHONPATH="/lus/flare/projects/alcf_training/io-libraries/soft/lib/python3.10/site-packages/:${PYTHONPATH}"
#export PYTHONPATH="/lus/flare/projects/alcf_training/io-libraries/soft/pydarshan-3.4.7-patched/lib/python3.10/site-packages/:$PYTHONPATH"
