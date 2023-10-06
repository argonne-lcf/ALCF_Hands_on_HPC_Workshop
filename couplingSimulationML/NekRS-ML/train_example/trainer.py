# General imports
import sys
import os
import argparse
import logging
import numpy as np
from time import sleep, perf_counter
from os.path import exists

# ML imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Distributed training imports
from mpi4py import MPI
import horovod.torch as hvd
from horovod.torch.mpi_ops import Sum

# SmartRedis imports
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
def init_client(SSDB, args, logger_init):
    if (args.dbnodes==1):
        tic = perf_counter()
        client = Client(address=SSDB,cluster=False)
        toc = perf_counter()
    else:
        tic = perf_counter()
        client = Client(address=SSDB,cluster=True)
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
    # Dataset that generates a key for DB tensor with varying rank ID
    # but fixed time step number
    def __init__(self, num_tot_tensors, step_num, head_rank):
        self.total_data = num_tot_tensors
        self.step = step_num
        self.head_rank = head_rank

    def __len__(self):
        return self.total_data

    def __getitem__(self, idx):
        tensor_num = idx+self.head_rank
        return f"x.{tensor_num}.{self.step}"
    
class RankStepDataset(torch.utils.data.Dataset):
    # Dataset that generates a key for DB tensor with varying rank ID
    # and varying time step number
    def __init__(self, num_ranks, steps, head_rank):
        self.ranks = num_ranks
        self.steps = steps
        self.num_steps = len(steps)
        self.head_rank = head_rank
        self.total_data = self.ranks*self.num_steps

    def __len__(self):
        return self.total_data

    def __getitem__(self, idx):
        rank_id = idx%self.ranks
        rank_id = rank_id+self.head_rank
        step_id = idx//self.ranks
        step = self.steps[step_id]
        return f"x.{rank_id}.{step}"

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
          batch, ndIn, client, args, logger_data):
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
        train_loader = DataLoader(mbdata, shuffle=True, batch_size=batch)
        for batch_idx, dbdata in enumerate(train_loader):
            # split inputs and outputs
            if (args.device != 'cpu'):
               dbdata = dbdata.to(args.device)
            features = dbdata[:, :ndIn]
            target = dbdata[:, ndIn:]

            optimizer.zero_grad()
            output = model.forward(features)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if ((batch_idx)%10==0):
                print(f'Train Epoch: {epoch} | ' + \
                      f'[{tensor_idx+1}/{len(train_tensor_loader)}] | ' + \
                      f'[{batch_idx+1}/{len(train_loader)}] | ' + \
                      f'Loss: {loss.item():>8e}')
                sys.stdout.flush()

    running_loss = running_loss / len(train_loader) / len(train_tensor_loader)
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
    # MPI import and initialization
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    print(f'Rank {rank}/{size} says hello from {name}')
    comm.Barrier()
    sys.stdout.flush()

    # Horovod import and initialization
    hvd.init()
    hrank = hvd.rank()
    hrankl = hvd.local_rank()
    hsize = hvd.size()
    print(f'HVD rank {hrank}/{hsize} and local rank {hrankl} says hello')
    comm.Barrier()
    sys.stdout.flush()

    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dbnodes',default=1,type=int,help='Number of database nodes')
    parser.add_argument('--device',default='cpu',help='Device to run on')
    parser.add_argument('--ppn',default=1,type=int,help='Number of processes per node')
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
    address = os.environ['SSDB']
    client = init_client(address, args, logger_init)
    comm.Barrier()
    if (rank == 0):
        print("All Python clients initialized\n")
        sys.stdout.flush()

    # Pull metadata from database
    while True:
        if (client.poll_tensor("tensorInfo",0,1)):
            tic = perf_counter()
            dataSizeInfo = client.get_tensor('tensorInfo').astype('int32')
            toc = perf_counter()
            time_meta = time_meta + (toc-tic)
            break
    comm.Barrier()
    npts = dataSizeInfo[0]
    num_tot_tensors = dataSizeInfo[1]
    num_db_tensors = dataSizeInfo[2]
    head_rank = dataSizeInfo[3]
    ndIn = dataSizeInfo[4]
    ndOut = dataSizeInfo[5]
    if (rank == 0):
        print("Retreived metadata from DB:")
        print(f"Number of samples per tensor: {npts}")
        print(f"Number of total tensors in all DB: {num_tot_tensors}")
        print(f"Number of tensors in local DB: {num_db_tensors}")
        print(f"Number of inputs and outputs to model: {ndIn},{ndOut}")
    sys.stdout.flush()

    # NN Training Hyper-Parameters
    Nepochs = 100 # number of epochs
    batch =  int(num_db_tensors/args.ppn) # how many tensors to grab from db
    mini_batch = 128 # batch size once tensors obtained from db and concatenated 
    learning_rate = 0.001 # learning rate
    nNeurons = 20 # number of neuronsining settings
    tol = 1.0e-7 # convergence tolerance on loss function

    # Set device to run on
    if (rank == 0):
        print(f"\nRunning on device: {args.device} \n")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if (args.device == 'cuda'):
        if torch.cuda.is_available():
            device_id = hrankl if torch.cuda.device_count()>1 else 0
            torch.cuda.set_device(device_id)

    # Instantiate the NN model and optimizer
    model = NeuralNetwork(inputDim=ndIn, outputDim=ndOut, numNeurons=nNeurons)
    if (args.device != 'cpu'):
        model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate*size)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         op=Sum)

    # Training setup and variable initialization
    istep = -1 # initialize the simulation step number to -1
    step_list = [] # initialize an empty list containing all the steps sent
    iepoch = 1 # epoch number

    # While loop that checks when training data is available on database
    if (rank == 0):
        print("Starting training loop ... \n")
    while True:
        # check to see if the time step number has been sent to database, if not cycle
        if (client.poll_tensor("step",0,1)):
            tic = perf_counter()
            tmp = client.get_tensor('step').astype('int32')
            toc = perf_counter()
            time_meta = time_meta + (toc-tic)
        else:
            continue

        # new data is available in database so update Dataset and DataLoader
        if (istep != tmp[0]): 
            istep = tmp[0]
            step_list.append(istep)
            batch =  int(num_db_tensors*len(step_list)/args.ppn)
            if (rank == 0):
                print("\nGetting new training data from DB ...")
                print(f"Added time step {istep} to training data\n")

            datasetTrain = RankStepDataset(num_db_tensors,step_list,head_rank)
            train_sampler = DistributedSampler(datasetTrain, num_replicas=args.ppn, 
                                               rank=hrankl, drop_last=False)
            train_tensor_loader = DataLoader(datasetTrain, batch_size=batch, 
                                             sampler=train_sampler)
        
        if (rank == 0):
            print(f"\n Epoch {iepoch}\n-------------------------------")
        
        model, global_loss = train(model, train_sampler, train_tensor_loader, optimizer,
                                    iepoch, mini_batch, ndIn, client, args, logger_data)
            
        # check if tolerance on loss is satisfied
        if (global_loss <= tol):
            if (rank == 0):
                print("Convergence tolerance met. Stopping training loop. \n")
            break
        
        # check if max number of epochs is reached
        if (iepoch >= Nepochs):
            if (rank == 0):
                print("Max number of epochs reached. Stopping training loop. \n")
            break

        iepoch = iepoch + 1        
        sys.stdout.flush()
        
        # new data is not avalable, so train another epoch on current data
        """
        else:
            if (rank == 0):
                print(f"\n Epoch {iepoch}\n-------------------------------")
        
            model, global_loss = train(model, train_sampler, train_tensor_loader, optimizer,
                                       iepoch, mini_batch, ndIn, client, args, logger_data)
            
            # check if tolerance on loss is satisfied
            if (global_loss <= tol):
                if (rank == 0):
                    print("Convergence tolerance met. Stopping training loop. \n")
                break
        
            # check if max number of epochs is reached
            if (iepoch >= Nepochs):
                if (rank == 0):
                    print("Max number of epochs reached. Stopping training loop. \n")
                break

            iepoch = iepoch + 1        
            sys.stdout.flush()
        """

    if (args.logging=='verbose'):
        logger_meta.info('%.8e',time_meta)    

    # Save model to file before exiting
    if (rank == 0):
        model.double()
        model_name = "model"
        torch.save(model.state_dict(), f"{model_name}.pt", _use_new_zipfile_serialization=False)
        # save jit traced model to be used for online inference with SmartSim
        features = np.double(np.random.uniform(low=0, high=10, size=(npts,ndIn)))
        features = torch.from_numpy(features).to(args.device)
        module = torch.jit.trace(model, features)
        torch.jit.save(module, f"{model_name}_jit.pt")
        print("Saved model to disk\n")
        sys.stdout.flush()


    # Exit and tell data loader to exit too
    comm.Barrier()
    if (rank%args.ppn == 0):
        print(f"[{hrank}]: Telling NekRS to quit ... \n")
        arrMLrun = np.int32(np.zeros(2))
        client.put_tensor("check-run",arrMLrun)

    if (rank==0):
        print("Exiting ...")
    

###
if __name__ == '__main__':
    main()
