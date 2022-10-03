# general imports
import os
import sys

# smartsim and smartredis imports
from smartsim import Experiment
from smartsim.settings import MpiexecSettings

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
ml_nNodes = int(sys.argv[4])
ppn = int(sys.argv[5])
simprocs = int(sys.argv[6])
simprocs_pn = int(sys.argv[7])
mlprocs = int(sys.argv[8])
mlprocs_pn = int(sys.argv[9])
device = sys.argv[10]
logging = sys.argv[11]
hostfile = sys.argv[12]

# Get nodes of this allocation (job) and split them between the tasks
nodelist, nNodes = parseNodeList(hostfile)
print(f"\nRunning on {nNodes} total nodes on Polaris")
print(nodelist, "\n")
dbNodes = ','.join(nodelist[0: db_nNodes])
dbNodes_list = nodelist[0: db_nNodes]
simNodes = ','.join(nodelist[db_nNodes: db_nNodes + sim_nNodes])
mlNodes = ','.join(nodelist[db_nNodes + sim_nNodes: db_nNodes + sim_nNodes + ml_nNodes])
print(f"Database running on {db_nNodes} nodes:")
print(dbNodes)
print(f"Simulatiom running on {sim_nNodes} nodes:")
print(simNodes)
print(f"ML running on {ml_nNodes} nodes:")
print(mlNodes, "\n")

# Set up database and start it
PORT = 6780
exp = Experiment("train-example", launcher="pbs")
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

# Fortran data producer
print("Launching Fortran data producer ...")
Ftn_exe = './src/dataLoaderFtn.exe'
if (simprocs_pn>ppn):
    simprocs_pn = ppn
runArgs = {"np": simprocs, "ppn": simprocs_pn, 
           "hosts": simNodes
          }
run_settings = MpiexecSettings(
               Ftn_exe,
               exe_args=None,
               run_args=runArgs
               )
sim_exp = exp.create_model("load_data", run_settings)
exp.start(sim_exp, summary=True, block=False)
print("Done\n")

# Python data consumer (i.e. training)
print("Launching Pyhton data consumer ...")
ml_exe = "src/trainPar.py"
if (device!='cpu' and mlprocs_pn>4):
    ppn = 4
else:
    ppn = mlprocs_pn
exe_args = ml_exe + f' --dbnodes {db_nNodes} --device {device} --ppn {mlprocs_pn} --logging {logging}'
runArgs = {"np": mlprocs, "ppn": mlprocs_pn, 
           "hosts": mlNodes
          }
runML_settings = MpiexecSettings('python',
                  exe_args=exe_args, 
                  run_args=runArgs
                  )
ml_exp = exp.create_model("train_model", runML_settings)
exp.start(ml_exp, summary=True, block=True)
print("Done\n")

# Stop database
print("Stopping the Orchestrator ...")
exp.stop(db)
print("Done")
print("Quitting")

