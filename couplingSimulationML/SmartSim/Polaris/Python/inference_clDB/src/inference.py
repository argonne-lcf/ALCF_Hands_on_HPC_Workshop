import sys
import argparse
from time import sleep, perf_counter
import numpy as np
import logging
from smartredis import Client

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file,mode='w')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def init_client(args, logger_init):
    if (args.dbnodes==1):
        tic = perf_counter()
        client = Client(cluster=False)
        toc = perf_counter()
    else:
        tic = perf_counter()
        client = Client(cluster=True)
        toc = perf_counter()
    if (args.logging=='verbose'):
        logger_init.info('%.8e',toc-tic)
    return client

def main():
    # Import and initialize MPI
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    print(f'Rank {rank}/{size} says hello from node {name}') 
    comm.Barrier()
    sys.stdout.flush()

    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dbnodes',default=1,type=int,help='Number of database nodes')
    parser.add_argument('--device',default='cpu',help='Device to run on')
    parser.add_argument('--ppn',default=2,type=int,help='Number of processes per node')
    parser.add_argument('--logging',default='no',help='Level of performance logging')
    args = parser.parse_args()

    # Create log files
    if (args.logging=='verbose'):
        logger_init = setup_logger('client_init', f'client_init_{rank}.log')
        logger_model = setup_logger('model', f'model_upload_{rank}.log')
        logger_inf = setup_logger('inference', f'inference_{rank}.log')
    else:
        logger_init = None

    # Initialize SmartRedis clients
    client = init_client(args, logger_init)
    comm.Barrier()
    if (rank==0):
        print('All SmartRedis clients initialized')
        sys.stdout.flush()

    # Load model onto Orchestrator
    tic = 0.; toc = 0.
    if (rank==0):
        device_tag = 'CPU' if args.device=='cpu' else 'GPU'
        tic = perf_counter()
        client.set_model_from_file('model', './model_jit.pt', 'TORCH', device=device_tag)
        toc = perf_counter()
        print(f'Uploaded model to database {device_tag} from rank {rank}')
    if (args.logging=='verbose'):
        logger_model.info('%.8e',toc-tic)
    comm.Barrier()
    sys.stdout.flush()

    # Set parameters for array of random numbers to be set as inference data
    # In this example we create inference data for a simple function
    # y=f(x), which has 1 input (x) and 1 output (y)
    # The domain for the function is from 0 to 10
    # The inference data is obtained from a uniform distribution over the domain
    nSamples = 1024
    xmin = 0.0 
    xmax = 10.0

    # Generate the key for the inference data
    # The key will be tagged with the rank ID
    inf_key = 'x.'+str(rank)
    pred_key = 'p.'+str(rank)

    # Open file to write predictions
    if (rank==0):
        fid = open('./predictions.dat', 'w')

    # Emulate integration of PDEs with a do loop
    numts = 5
    for its in range(numts):
        sleep(10)

        # Generate the input data for the polynomial y=f(x)=x**2 + 3*x + 1
        inputs = np.random.uniform(low=xmin, high=xmax, size=(nSamples,1))

        # Perform inferece
        tic_s = perf_counter()
        client.put_tensor(inf_key, inputs)
        toc_s = perf_counter()
        tic_i = perf_counter()
        client.run_model('model', inputs=[inf_key], outputs=[pred_key])
        toc_i = perf_counter()
        tic_r = perf_counter()
        predictions = client.get_tensor(pred_key)
        toc_r = perf_counter()
        if (args.logging=='verbose'):
            logger_inf.info('%.8e %.8e %.8e',toc_s-tic_s,toc_i-tic_i,toc_r-tic_r)
        comm.Barrier()
        if (rank==0):
            print(f'Performed inference on all ranks for step {its+1}')
            sys.stdout.flush()

        # Write predictions to file
        if (rank==0):
            truth = inputs**2 + 3*inputs + 1
            for i in range(nSamples):
                fid.write(f'{inputs[i,0]:.6e} {predictions[i,0]:.6e} {truth[i,0]:.6e}\n')

    if (rank==0):
        fid.close()
    
    if (rank==0):
        print('Exiting ...')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
