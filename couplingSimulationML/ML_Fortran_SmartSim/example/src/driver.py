import os, sys, time
from smartsim import Experiment
from smartsim.settings import AprunSettings
from smartsim.database import CobaltOrchestrator

# Parse command line arguments
ppn = int(sys.argv[1])
nodes = int(sys.argv[2])
allprocs = int(sys.argv[3])
dbnodes = int(sys.argv[4])
simnodes = int(sys.argv[5])
mlnodes = int(sys.argv[6])
simprocs = int(sys.argv[7])
mlprocs = int(sys.argv[8])

assert (dbnodes + simnodes + mlnodes <= nodes) and (mlnodes >= 0), "The nodes requested are not enough."



# Define function to parse node list
def parseNodeList():
    cobStr = os.environ['COBALT_PARTNAME']
    tmp = cobStr.split(',')
    nodelist = []
    for item in tmp:
        if (item.find('-') > 0):
            tmp2 = item.split('-')
            istart = int(tmp2[0])
            iend = int(tmp2[1])
            for i in range(istart,iend+1):
                nodelist.append(str(i))
        else:
            nodelist.append(item)
    nnodes = len(nodelist)
    return nodelist, nnodes

# Get nodes of this allocation (job) and split them between the tasks
nodelist, nnodes = parseNodeList()
print(f"\nRunning on {nnodes} total nodes on Theta")
print(nodelist, "\n")
dbNodes = ','.join(nodelist[0: dbnodes])
simNodes = ','.join(nodelist[dbnodes: dbnodes + simnodes])
mlNodes = ','.join(nodelist[dbnodes + simnodes: dbnodes + simnodes + mlnodes])
print(f"Database running on {dbnodes} nodes:")
print(dbNodes)
print(f"Simulatiom running on {simnodes} nodes:")
print(simNodes)
print(f"ML running on {mlnodes} nodes:")
print(mlNodes, "\n")

# Set up database and start it
PORT = 6780
exp = Experiment("train-example", launcher="cobalt")
db = CobaltOrchestrator(port=PORT, batch=False, db_nodes=dbnodes, run_args={"node-list": dbNodes})
print("Starting database ...")
exp.start(db)
print("Done\n")

# data producer
print("Launching data producer ...")
Ftn_exe = 'src/dataLoaderFtn'
aprun = AprunSettings(Ftn_exe, run_args={"node-list": simNodes})
aprun.set_tasks(simprocs)
aprun.set_tasks_per_node(ppn)
load_data = exp.create_model("load_data", aprun)
exp.start(load_data, summary=False, block=False)
print("Done\n")

# data consumer
print("Launching data consumer ...")
ml_exe = "src/trainPar.py"
aprunML = AprunSettings("python", 
        exe_args=ml_exe, 
        run_args={"node-list": mlNodes})
aprunML.set_tasks(mlprocs)
aprunML.set_tasks_per_node(mlprocs)
ml_model = exp.create_model("train_model", aprunML)
exp.start(ml_model, summary=False, block=True)
print("Done\n")


# Stop database
print("Stopping the Orchestrator ...")
exp.stop(db)
print("Done")
print("Quitting")

