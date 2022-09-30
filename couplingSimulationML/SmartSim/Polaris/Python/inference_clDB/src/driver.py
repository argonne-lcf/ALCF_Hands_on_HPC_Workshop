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
db_nNodes = int(sys.argv[2])
sim_nNodes = int(sys.argv[3])
ppn = int(sys.argv[4])
simprocs = int(sys.argv[5])
simprocs_pn = int(sys.argv[6])
device = sys.argv[7]
logging = sys.argv[8]
hostfile = sys.argv[9]

# Get nodes of this allocation (job) and split them between the tasks
nodelist, nNodes = parseNodeList(hostfile)
print(f"\nRunning on {nNodes} total nodes on Polaris")
print(nodelist, "\n")
dbNodes = ','.join(nodelist[0: db_nNodes])
dbNodes_list = nodelist[0: db_nNodes]
simNodes = ','.join(nodelist[db_nNodes: db_nNodes + sim_nNodes])
print(f"Database running on {db_nNodes} nodes:")
print(dbNodes)
print(f"Simulatiom running on {sim_nNodes} nodes:")
print(simNodes)
print("")

# Set up database and start it
PORT = 6780
exp = Experiment("inference-example", launcher="pbs")
runArgs = {"np": 1, "ppn": ppn, 
           #"hosts": dbNodes
          }
db = exp.create_database(port=PORT, batch=False, db_nodes=db_nNodes,
                         run_command='mpiexec',
                         interface='hsn0', 
                         hosts=dbNodes_list,
                         run_args=runArgs,
                         single_cmd=False
                        )
exp.generate(db)
print("Starting database ...")
exp.start(db)
print("Done\n")

# Python inference routine
print("Launching Python inference routine ...")
Py_exe = './src/inference.py'
if (simprocs_pn>ppn):
    simprocs_pn = ppn
exe_args = Py_exe + f' --dbnodes {db_nNodes} --device {device} --ppn {simprocs_pn} --logging {logging}'
runArgs = {"np": simprocs, "ppn": simprocs_pn, 
           "hosts": simNodes
          }
run_settings = MpiexecSettings('python',
                  exe_args=exe_args, 
                  run_args=runArgs
                  )
inf_exp = exp.create_model("inference", run_settings)
exp.start(inf_exp, summary=True, block=True)
print("Done\n")

# Stop database
print("Stopping the Orchestrator ...")
exp.stop(db)
print("Done")
print("Quitting")

