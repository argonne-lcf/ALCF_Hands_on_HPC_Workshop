# general imports
import os
import sys 
from omegaconf import DictConfig, OmegaConf
import hydra

# smartsim and smartredis imports
from smartsim import Experiment
from smartsim.settings import RunSettings, PalsMpiexecSettings


## Define function to parse node list
def parseNodeList(fname):
    with open(fname) as file:
        nodelist = file.readlines()
        nodelist = [line.rstrip() for line in nodelist]
        nodelist = [line.split('.')[0] for line in nodelist]
    nNodes = len(nodelist)
    return nodelist, nNodes


## Co-located DB launch
def launch_coDB(cfg, nodelist, nNodes):
    # Print nodelist
    if (nodelist is not None):
        print(f"\nRunning on {nNodes} total nodes on Polaris")
        print(nodelist, "\n")
        hosts = ','.join(nodelist)

    # Initialize the SmartSim Experiment
    PORT = cfg.database.port
    exp = Experiment(cfg.database.exp_name, launcher=cfg.database.launcher)

    # Set the run settings, including the client executable and how to run it
    client_exe = cfg.sim.executable
    if (cfg.database.launcher=='local'):
        nrs_settings = RunSettings(client_exe,
                           exe_args=cfg.sim.arguments,
                           run_command='mpirun',
                           run_args={"-n" : cfg.run_args.simprocs},
                           env_vars=None)
    elif (cfg.database.launcher=='pbs'):
        nrs_settings = PalsMpiexecSettings(
                           client_exe,
                           exe_args=cfg.sim.arguments,
                           run_args=None,
                           env_vars=None)
        nrs_settings.set_tasks(cfg.run_args.simprocs)
        nrs_settings.set_tasks_per_node(cfg.run_args.simprocs_pn)
        nrs_settings.set_hostlist(hosts)
        nrs_settings.set_cpu_binding_type(cfg.run_args.sim_cpu_bind)
        if (cfg.sim.affinity):
            nrs_settings.set_gpu_affinity_script(cfg.sim.affinity,
                                                 cfg.run_args.simprocs_pn)

    # Create the co-located database model
    colo_model = exp.create_model("nekrs", nrs_settings)
    kwargs = {
        'maxclients': 100000,
        'threads_per_queue': 4, # set to 4 for improved performance
        'inter_op_parallelism': 1,
        'intra_op_parallelism': 1,
        'cluster-node-timeout': 30000,
        }
    if (cfg.database.backend == "keydb"):
        kwargs['server_threads'] = 2 # keydb only
    if (cfg.database.network_interface=='uds'):
        colo_model.colocate_db_uds(
                db_cpus=cfg.run_args.dbprocs_pn,
                debug=False,
                limit_app_cpus=True,
                **kwargs
                )
    else:
        colo_model.colocate_db(
                port=PORT,
                db_cpus=cfg.run_args.dbprocs_pn,
                debug=False,
                limit_app_cpus=True,
                ifname=cfg.database.network_interface,
                **kwargs
                )
    
    # Load a model for inference
    if (cfg.inference.model_path):
        colo_model.add_ml_model('model',
                                cfg.inference.backend,
                                model=None,  # this is used if model is in memory
                                model_path=cfg.inference.model_path,
                                device=cfg.inference.device,
                                batch_size=cfg.inference.batch,
                                min_batch_size=cfg.inference.batch,
                                devices_per_node=cfg.inference.devices_per_node,
                                inputs=None, outputs=None)

    # Start the co-located model
    block = False if cfg.train.executable else True
    print("Launching NekRS and SmartSim co-located DB ... ")
    exp.start(colo_model, block=block, summary=False)
    print("Done\n")

    # Setup and launch the training script
    if (cfg.train.executable):
        ml_exe = cfg.train.executable
        ml_exe = ml_exe + f' --dbnodes={cfg.run_args.db_nodes}' \
                        + f' --device={cfg.train.device}' \
                        + f' --ppn={cfg.run_args.mlprocs_pn}' \
                        + f' --logging={cfg.train.logging}'
        if (cfg.database.launcher=='local'):
            ml_settings = RunSettings(
                           'python',
                           exe_args=ml_exe,
                           run_command='mpirun',
                           run_args={"-n" : cfg.run_args.mlprocs},
                           env_vars=colo_model.run_settings.env_vars)
        elif (cfg.database.launcher=='pbs'):
            ml_settings = PalsMpiexecSettings(
                           'python',
                           exe_args=ml_exe,
                           run_args=None,
                           env_vars=colo_model.run_settings.env_vars)
            ml_settings.set_tasks(cfg.run_args.mlprocs)
            ml_settings.set_tasks_per_node(cfg.run_args.mlprocs_pn)
            ml_settings.set_hostlist(hosts)
            ml_settings.set_cpu_binding_type(cfg.run_args.ml_cpu_bind)
            if (cfg.train.affinity):
                ml_settings.set_gpu_affinity_script(cfg.train.affinity,
                                                    cfg.run_args.mlprocs_pn,
                                                    cfg.run_args.simprocs_pn)
        
        ml_model = exp.create_model("train_model", ml_settings)
        print("Launching training script ... ")
        exp.start(ml_model, block=True, summary=False)
        print("Done\n")


## Clustered DB launch
def launch_clDB(cfg, nodelist, nNodes):
    # Split nodes between the components
    dbNodes_list = None
    if (nodelist is not None):
        dbNodes = ','.join(nodelist[0: cfg.run_args.db_nodes])
        dbNodes_list = nodelist[0: cfg.run_args.db_nodes]
        simNodes = ','.join(nodelist[cfg.run_args.db_nodes: \
                                 cfg.run_args.db_nodes + cfg.run_args.sim_nodes])
        mlNodes = ','.join(nodelist[cfg.run_args.db_nodes + cfg.run_args.sim_nodes: \
                                cfg.run_args.db_nodes + cfg.run_args.sim_nodes + \
                                cfg.run_args.ml_nodes])
        print(f"Database running on {cfg.run_args.db_nodes} nodes:")
        print(dbNodes)
        print(f"Simulatiom running on {cfg.run_args.sim_nodes} nodes:")
        print(simNodes)
        print(f"ML running on {cfg.run_args.ml_nodes} nodes:")
        print(mlNodes, "\n")

    # Set up database and start it
    PORT = cfg.database.port
    exp = Experiment(cfg.database.exp_name, launcher=cfg.database.launcher)
    runArgs = {"np": 1, "ppn": 1, "cpu-bind": "numa"}
    kwargs = {
        'maxclients': 100000,
        'threads_per_queue': 4, # set to 4 for improved performance
        'inter_op_parallelism': 1,
        'intra_op_parallelism': cfg.run_args.cores_pn,
        'cluster-node-timeout': 30000,
        }
    if (cfg.database.backend == "keydb"):
        kwargs['server_threads'] = 2 # keydb only
    if (cfg.database.launcher=='local'): run_command = 'mpirun'
    elif (cfg.database.launcher=='pbs'): run_command = 'mpiexec'
    db = exp.create_database(port=PORT, 
                             batch=False,
                             db_nodes=cfg.run_args.db_nodes,
                             run_command=run_command,
                             interface=cfg.database.network_interface, 
                             hosts=dbNodes_list,
                             run_args=runArgs,
                             single_cmd=True,
                             **kwargs
                            )
    exp.generate(db)
    print("Starting database ...")
    exp.start(db)
    print("Done\n")

    # Set the run settings, including the client executable and how to run it
    client_exe = cfg.sim.executable
    if (cfg.database.launcher=='local'):
        run_settings = RunSettings(client_exe,
                                   exe_args=None,
                                   run_command='mpirun',
                                   run_args={"-n" : cfg.run_args.simprocs},
                                   env_vars=None)
    elif (cfg.database.launcher=='pbs'):
        run_settings = PalsMpiexecSettings(client_exe,
                                           exe_args=None,
                                           run_args=None,
                                           env_vars=None)
        run_settings.set_tasks(cfg.run_args.simprocs)
        run_settings.set_tasks_per_node(cfg.run_args.simprocs_pn)
        run_settings.set_hostlist(simNodes)
        run_settings.set_cpu_binding_type(cfg.run_args.sim_cpu_bind)
    client_exp = exp.create_model("client", run_settings)

    # Start the client model
    print("Launching the client ...")
    block = False if cfg.train.executable else True
    exp.start(client_exp, summary=False, block=block)
    print("Done\n")
    
    # Setup and launch the training script
    if (cfg.train.executable):
        ml_exe = cfg.train.executable
        ml_exe = ml_exe + f' --dbnodes={cfg.run_args.db_nodes}' \
                        + f' --device={cfg.train.device}' \
                        + f' --ppn={cfg.run_args.mlprocs_pn}' \
                        + f' --logging={cfg.train.logging}'
        if (cfg.database.launcher=='local'):
            ml_settings = RunSettings(
                           'python',
                           exe_args=ml_exe,
                           run_command='mpirun',
                           run_args={"-n" : cfg.run_args.mlprocs},
                           env_vars=None)
        elif (cfg.database.launcher=='pbs'):
            ml_settings = PalsMpiexecSettings(
                           'python',
                           exe_args=ml_exe,
                           run_args=None,
                           env_vars=None)
            ml_settings.set_tasks(cfg.run_args.mlprocs)
            ml_settings.set_tasks_per_node(cfg.run_args.mlprocs_pn)
            ml_settings.set_hostlist(mlNodes)
            ml_settings.set_cpu_binding_type(cfg.run_args.ml_cpu_bind)
        
        ml_model = exp.create_model("train_model", ml_settings)
        print("Launching training script ... ")
        exp.start(ml_model, block=True, summary=False)
        print("Done\n")
    
    # Stop database
    print("Stopping the Orchestrator ...")
    exp.stop(db)
    print("Done\n")


## Main function
@hydra.main(version_base=None, config_path="./conf", config_name="ssim_config")
def main(cfg: DictConfig):
    # Get nodes of this allocation (job)
    nodelist = nNodes = None
    if (cfg.database.launcher=='pbs'):
        hostfile = os.getenv('PBS_NODEFILE')
        nodelist, nNodes = parseNodeList(hostfile)

    # Export KeyDB env variables if requested instead of Redis
    if (cfg.database.backend == "keydb"):
        stream = os.popen("which python")
        prefix = stream.read().split("ssim")[0]
        keydb_conf = prefix+"SmartSim/smartsim/_core/config/keydb.conf"
        keydb_bin = prefix+"keydb/src/keydb-server"
        os.environ['REDIS_PATH'] = keydb_bin
        os.environ['REDIS_CONF'] = keydb_conf

    # Call appropriate launcher
    if (cfg.database.deployment == "colocated"):
        print(f"\nRunning {cfg.database.deployment} DB with {cfg.database.backend} backend\n")
        launch_coDB(cfg, nodelist, nNodes)
    elif (cfg.database.deployment == "clustered"):
        print(f"\nRunning {cfg.database.deployment} DB with {cfg.database.backend} backend\n")
        launch_clDB(cfg, nodelist, nNodes)
    else:
        print("\nERROR: Launcher is either colocated or clustered\n")

    # Quit
    print("Quitting")


## Run main
if __name__ == "__main__":
    main()
