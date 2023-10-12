# load darshan module using MODULEPATH for now
export MODULEPATH=/soft/perftools/darshan/darshan-3.4.3/share/craype-2.x/modulefiles/:$MODULEPATH
module add darshan

# load a more recent Python version
module add cray-python

# point to PyDarshan install co-located with Darshan install
export PYTHONPATH=/soft/perftools/darshan/darshan-3.4.3/python
