import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from time import sleep, perf_counter
from os.path import exists
import horovod.torch as hvd

from smartredis import Client

## Define logger
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file,mode='w')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

## Initialize Redis clients
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

## Define the Neural Network Structure
class NeuralNetwork(nn.Module):
    # The class takes as inputs the input and output dimensions and the number of layers   
    def __init__(self, inputDim, outputDim, numNeurons):
        super().__init__()
        self.ndIn = inputDim
        self.ndOut = outputDim
        self.nNeurons = numNeurons
        self.net = nn.Sequential(
            nn.Linear(self.ndIn, self.nNeurons),
            nn.ReLU(),
            nn.Linear(self.nNeurons, self.nNeurons),
            nn.ReLU(),
            nn.Linear(self.nNeurons, self.nNeurons),
            nn.ReLU(),
            nn.Linear(self.nNeurons, self.ndOut),
        )

    # Define the method to do a forward pass
    def forward(self, x):
        return self.net(x)


## Define Datasets
class RankDataset(torch.utils.data.Dataset):
    # contains the keys of all tensors uploaded to db by phasta ranks
    def __init__(self, num_tot_tensors, step_num, head_rank):
        self.total_data = num_tot_tensors
        self.step = step_num
        self.head_rank = head_rank

    def __len__(self):
        return self.total_data

    def __getitem__(self, idx):
        tensor_num = idx+self.head_rank
        return f"y.{tensor_num}.{self.step}"

class MinibDataset(torch.utils.data.Dataset):
    #dataset of each ML rank in one epoch with the concatenated tensors
    def __init__(self,concat_tensor):
        self.concat_tensor = concat_tensor

    def __len__(self):
        return len(self.concat_tensor)

    def __getitem__(self, idx):
        return self.concat_tensor[idx]


## Training subroutine
def train(model, train_sampler, train_tensor_loader, optimizer, epoch, 
          batch, ndOut, client, args, logger_data):
    model.train()
    running_loss = 0.0
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)

    loss_fn = nn.functional.mse_loss

    for tensor_idx, tensor_keys in enumerate(train_tensor_loader):
        # grab data from database
        print(f'[{hvd.rank()}]: Grabbing tensors with key {tensor_keys}')
        hvd.allreduce(torch.tensor(0), name='barrier')
        sys.stdout.flush()
        tic = perf_counter()
        concat_tensor = torch.cat([torch.from_numpy(client.get_tensor(key)) \
                      for key in tensor_keys], dim=0)
        toc = perf_counter()
        if (args.logging=='verbose'):
            logger_data.info('%.8e',toc-tic)
        concat_tensor = concat_tensor.float()

        mbdata = MinibDataset(concat_tensor)
        train_loader = torch.utils.data.DataLoader(mbdata, shuffle=True, batch_size=batch)
        for batch_idx, dbdata in enumerate(train_loader):
            # split inputs and outputs
            if (args.device != 'cpu'):
               dbdata = dbdata.to(args.device)
            target = dbdata[:, :ndOut]
            features = dbdata[:, ndOut:]

            optimizer.zero_grad()
            output = model.forward(features)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    running_loss = running_loss / len(train_loader) / batch
    loss_avg = metric_average(running_loss, 'running_loss')

    if hvd.rank() == 0:
        print(f"Training set: Average loss: {loss_avg:>8e}")

    return model, loss_avg


## Average across hvd ranks
def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


## Main
def main():
    # Horovod import and initialization
    #import horovod.torch as hvd
    from horovod.torch.mpi_ops import Sum
    hvd.init()
    hrank = hvd.rank()
    hrankl = hvd.local_rank()
    hsize = hvd.size()
    print(f'HVD rank {hrank}/{hsize} and local rank {hrankl}')
    sys.stdout.flush()

    # MPI import and initialization
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    print(f'Rank {rank}/{size} says hello from node {name}')
    sys.stdout.flush()

    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dbnodes',default=1,type=int,help='Number of database nodes')
    parser.add_argument('--device',default='cpu',help='Device to run on')
    parser.add_argument('--ppn',default=2,type=int,help='Number of processes per node')
    parser.add_argument('--logging',default='no',help='Level of performance logging')
    args = parser.parse_args()

    # Create log files
    time_meta = 0.
    if (args.logging=='verbose'):
        logger_init = setup_logger('client_init', f'client_init_ml_{rank}.log')
        logger_meta = setup_logger('meta', f'meta_data_ml_{rank}.log')
        logger_data = setup_logger('train_data', f'train_data_ml_{rank}.log')
    else:
        logger_init = None
        logger_data = None

    # Initialize Redis clients on each rank
    client = init_client(args, logger_init)
    comm.Barrier()
    if (hrank == 0):
        print("All Python clients initialized\n")
        sys.stdout.flush()

    # Pull metadata from database
    while True:
        if (client.poll_tensor("sizeInfo",0,1)):
            tic = perf_counter()
            dataSizeInfo = client.get_tensor('sizeInfo')
            toc = perf_counter()
            time_meta = time_meta + (toc-tic)
            break
    comm.Barrier()
    if (hrank == 0):
        print("Retreived metadata from DB \n")

    npts = dataSizeInfo[0]
    ndTot = dataSizeInfo[1]
    ndIn = dataSizeInfo[2]
    ndOut = ndTot - ndIn
    num_tot_tensors = dataSizeInfo[3]
    num_db_tensors = dataSizeInfo[4]
    head_rank = dataSizeInfo[5]
    print(f'[{rank}]: head rank is {head_rank}')
    sys.stdout.flush()

    # NN Training Hyper-Parameters
    Nepochs = 100 # number of epochs
    batch =  int(num_db_tensors/args.ppn) # how many tensors to grab from db
    mini_batch = 16 # batch size once tensors obtained from db and concatenated 
    learning_rate = 0.001*hsize # learning rate
    nNeurons = 20 # number of neuronsining settings
    tol = 1.0e-1 # convergence tolerance on loss function

    # Set device to run on
    if (hrank == 0):
        print(f"Running on device: {args.device} \n")
    device = torch.device(args.device)
    if (args.device == 'cuda'):
        if torch.cuda.is_available():
            torch.cuda.set_device(hvd.local_rank())

    # Instantiate the NN model and optimizer
    model = NeuralNetwork(inputDim=ndIn, outputDim=ndOut, numNeurons=nNeurons)
    if (args.device != 'cpu'):
        model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training setup and variable initialization
    istep = 0 # initialize the simulation step number to 0
    iepoch = 1 # epoch number
    torch.set_num_threads(hsize)

    # While loop that checks when training data is available on database
    if (hrank == 0):
        print("Starting training loop ... \n")
    while True:
        # check to see if the time step number has been sent to database, if not cycle
        if (client.poll_tensor("step",0,1)):
            tic = perf_counter()
            tmp = client.get_tensor('step')
            toc = perf_counter()
            time_meta = time_meta + (toc-tic)
        else:
            continue

        # new data is available in database so update it and train 1 epoch
        if (istep != tmp[0]): 
            istep = tmp[0]
            if (hrank == 0):
                print("\nGetting new training data from DB ...")
                print(f"Working on time step {istep} \n")

            datasetTrain = RankDataset(num_db_tensors,istep,head_rank)
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                        datasetTrain, num_replicas=hsize, rank=hrank, drop_last=False)
            train_tensor_loader = torch.utils.data.DataLoader(
                        datasetTrain, batch_size=batch, sampler=train_sampler)
       
            if (iepoch==1):
                hvd.broadcast_parameters(model.state_dict(), root_rank=0)
                hvd.broadcast_optimizer_state(optimizer, root_rank=0)
                optimizer = hvd.DistributedOptimizer(
                                 optimizer,named_parameters=model.named_parameters(),op=Sum)
        
            if (hrank == 0):
                print(f"\n Epoch {iepoch}\n-------------------------------")
        
            model, global_loss = train(model, train_sampler, train_tensor_loader, optimizer,
                                       iepoch, batch, ndOut, client, args, logger_data)
            
            # check if tolerance on loss is satisfied
            if (global_loss <= tol):
                if (hrank == 0):
                    print("Convergence tolerance met. Stopping training loop. \n")
                break
        
            # check if max number of epochs is reached
            if (iepoch >= Nepochs):
                if (hrank == 0):
                    print("Max number of epochs reached. Stopping training loop. \n")
                break

            iepoch = iepoch + 1        
            sys.stdout.flush()
            sleep(0.5)
        
        # new data is not avalable, so train another epoch on current data
        else:
            if (hrank == 0):
                print(f"\n Epoch {iepoch}\n-------------------------------")
        
            model, global_loss = train(model, train_sampler, train_tensor_loader, optimizer,
                                       iepoch, batch, ndOut, client, args, logger_data)
            
            # check if tolerance on loss is satisfied
            if (global_loss <= tol):
                if (hrank == 0):
                    print("Convergence tolerance met. Stopping training loop. \n")
                break
        
            # check if max number of epochs is reached
            if (iepoch >= Nepochs):
                if (hrank == 0):
                    print("Max number of epochs reached. Stopping training loop. \n")
                break

            iepoch = iepoch + 1        
            sys.stdout.flush()
            sleep(0.5)

    if (args.logging=='verbose'):
        logger_meta.info('%.8e',time_meta)    

    # Save model to file before exiting
    if (hrank == 0):
        model_name = "model"
        torch.save(model.state_dict(), f"{model_name}.pt", _use_new_zipfile_serialization=False)
        # save jit traced model to be used for online inference with SmartSim
        features = np.float32(np.random.uniform(low=0, high=10, size=(100,1)))
        features = torch.from_numpy(features).to(args.device)
        module = torch.jit.trace(model, features)
        torch.jit.save(module, f"{model_name}_jit.pt")
        print("Saved model to disk\n")
        sys.stdout.flush()


    # Exit and tell data loader to exit too
    comm.Barrier()
    if (hrank == 0):
        print("Telling data loader to quit ... \n")
        arrMLrun = np.zeros(2)
        client.put_tensor("check-run",arrMLrun)

        print("Exiting ...")
    

###
if __name__ == '__main__':
    main()

