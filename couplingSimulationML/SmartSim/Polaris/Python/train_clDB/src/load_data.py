import sys
import os
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
    parser.add_argument('--ppn',default=4,type=int,help='Number of processes per node')
    parser.add_argument('--logging',default='no',help='Level of performance logging')
    args = parser.parse_args()
    if (rank==0):
        print(f'\nRunning with {args.dbnodes} DB nodes')
        print(f'and with {args.ppn} processes per node \n')
        sys.stdout.flush()

    # Create log files
    time_meta = 0.
    if (args.logging=='verbose'):
        logger_init = setup_logger('client_init', f'client_init_{rank}.log')
        logger_meta = setup_logger('meta', f'meta_data_{rank}.log')
        logger_data = setup_logger('train_data', f'train_data_{rank}.log')
    else:
        logger_init = None

    # Initialize SmartRedis clients
    client = init_client(args, logger_init)
    comm.Barrier()
    if (rank==0):
        print('All SmartRedis clients initialized')
        sys.stdout.flush()
    
    # Set parameters for array of random numbers to be set as inference data
    # In this example we create inference data for a simple function
    # y=f(x), which has 1 input (x) and 1 output (y)
    # The domain for the function is from 0 to 10
    # The inference data is obtained from a uniform distribution over the domain
    nSamples = 64
    xmin = 0.0 
    xmax = 10.0
    nInputs = 1
    nOutputs = 1

    # Send array used to communicate whether to keep running data loader or ML
    if (rank == 0):
        arrMLrun = np.array([1, 1])
        tic = perf_counter()
        client.put_tensor('check-run', arrMLrun)
        toc = perf_counter()
        time_meta = time_meta + (toc - tic)

    # Send some information regarding the training data size
    if (rank == 0):
        arrInfo = np.array([nSamples, nInputs+nOutputs, nInputs,
                            size, args.ppn, rank])
        tic = perf_counter()
        client.put_tensor('sizeInfo', arrInfo)
        toc = perf_counter()
        time_meta = time_meta + (toc - tic)
        print('Sent size info of training data to database')

    # Emulate integration of PDEs with a do loop
    numts = 1000
    stepInfo = np.zeros(2, dtype=int)
    for its in range(numts):
        # Sleep for a few seconds to emulate the time required by PDE integration
        sleep(2)

        # First off check if ML is done training, if so exit from loop
        tic = perf_counter()
        arrMLrun = client.get_tensor('check-run')
        toc = perf_counter()
        time_meta = time_meta + (toc - tic)
        if (arrMLrun[0]<0.5):
            break

        # Generate the training data for the polynomial y=f(x)=x**2 + 3*x + 1
        # place output in first column and input in second column
        inputs = np.random.uniform(low=xmin, high=xmax, size=(nSamples,1))
        outputs = inputs**2 + 3*inputs + 1
        sendArr = np.concatenate((outputs, inputs), axis=1)

        # Send training data to database
        send_key = 'y.'+str(rank)+'.'+str(its+1)
        if (rank==0):
            print(f'Sending training data with key {send_key} and shape {sendArr.shape}')
        tic = perf_counter()
        client.put_tensor(send_key, sendArr)
        toc = perf_counter()
        if (args.logging=='verbose'):
            logger_data.info('%.8e',toc-tic)
        comm.Barrier()
        if (rank==0):
            print(f'All ranks finished sending training data')

        # Send the time step number, used by ML program to determine
        # when new data is available
        if (rank == 0):
            stepInfo[0] = int(its+1)
            tic = perf_counter()
            client.put_tensor('step', stepInfo)
            toc = perf_counter()
            time_meta = time_meta + (toc - tic)

        sys.stdout.flush()

    if (args.logging=='verbose'):
        logger_meta.info('%.8e',time_meta)

    if (rank==0):
        print('Exiting ...')


if __name__ == '__main__':
    main()
