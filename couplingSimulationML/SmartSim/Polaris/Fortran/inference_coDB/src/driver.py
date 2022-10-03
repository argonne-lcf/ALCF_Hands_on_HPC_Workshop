# general imports
import os
import sys

# smartsim and smartredis imports
from smartsim import Experiment
from smartsim.settings import MpiexecSettings
from smartsim.settings import MpirunSettings

# Define function to parse node list
def parseNodeList(fname):
    with open(fname) as file:
        nodelist = file.readlines()
        nodelist = [line.rstrip() for line in nodelist]
        nodelist = [line.split('.')[0] for line in nodelist]
    nNodes = len(nodelist)
    return nodelist, nNodes

# Parse command line arguments
nodes = int(sys.argv[1])
ppn = int(sys.argv[2])
simprocs = int(sys.argv[3])
simprocs_pn = int(sys.argv[4])
dbprocs_pn = int(sys.argv[5])
device = sys.argv[6]
logging = sys.argv[7]
hostfile = sys.argv[8]

# Get nodes of this allocation (job) and split them between the tasks
nodelist, nNodes = parseNodeList(hostfile)
print(f"\nRunning on {nNodes} total nodes on Polaris")
print(nodelist, "\n")
hosts = ','.join(nodelist)

# Initialize the SmartSim Experiment
PORT = 6780
exp = Experiment("inference-example", launcher="pbs")

# Set the run settings, including the Fortran executable and how to run it
Ftn_exe = './src/inferenceFtn.exe'
if (simprocs_pn<ppn):
    ppn = simprocs_pn
runArgs = {'hostfile': hostfile,
           'np': simprocs, 'ppn': ppn,
          }
run_settings = MpiexecSettings(
               Ftn_exe,
               exe_args=None,
               run_args=runArgs,
               env_vars=None
               )

# Create and start the co-located database model
colo_model = exp.create_model("inference", run_settings)
#kwargs = {
#    maxclients: 100000,
#    threads_per_queue: 1, # set to 4 for improved performance
#    inter_op_threads: 1,
#    intra_op_threads: 1,
#    server_threads: 2 # keydb only
#}
colo_model.colocate_db(
        port=PORT,              # database port
        db_cpus=dbprocs_pn,     # cpus given to the database on each node
        debug=True,            # include debug information (will be slower)
        limit_app_cpus=False,    # limit the number of cpus used by the app
        ifname='lo',            # specify network interface to use (i.e. "ib0")
)
exp.start(colo_model, block=True, summary=True)

# Quit
print("Done")
print("Quitting")

