import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import time
import string
import random
import sys
from pathlib import Path
from time import sleep

from smartredis import Client

# Horovod import and initialization
import horovod.torch as hvd
from horovod.torch.mpi_ops import Sum
hvd.init()
hrank = hvd.rank()
hsize = hvd.size()

# MPI import and initialization
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


##### Initialize Redis clients on each rank #####
if (hrank == 0):
    print("\nInitializing Python clients ...")
nnDB = 1
if (nnDB==1):
    client = Client(cluster=False) # must change to True if running database on >1 nodes
else:
    client = Client(cluster=True)


##### Define Dataset class #####
class PhastaRankDataset(torch.utils.data.Dataset):
    # contains the keys of all tensors uploaded to db by phasta ranks
    def __init__(self, num_db_tensors, step_num):
        self.total_data = num_db_tensors
        self.step = step_num

    def __len__(self):
        return self.total_data

    def __getitem__(self, idx):
        return f"y.{idx}.{self.step}"

class MinibDataset(torch.utils.data.Dataset):
    #dataset of each ML rank in one epoch with the concatenated tensors
    def __init__(self,concat_tensor):
        self.concat_tensor = concat_tensor

    def __len__(self):
        return len(self.concat_tensor)

    def __getitem__(self, idx):
        return self.concat_tensor[idx]



###### Define the Neural Network Structure #####################
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


##### Pull metadata from database #####
# In this first case, data was split into the number of processes used for training
# so each rank here can take its data and the metadate
if (hrank == 0):
    print("\nGetting size info from DB ...\n")
while True:
    if (client.poll_tensor("sizeInfo",0,1)):
        dataSizeInfo = client.get_tensor('sizeInfo')
        break

npts = dataSizeInfo[0]
ndTot = dataSizeInfo[1]
ndIn = dataSizeInfo[2]
ndOut = ndTot - ndIn
num_db_tensors = dataSizeInfo[3]


##### NN Training Hyper-Parameters #####
Nepochs = 50 # number of epochs
batch =  int(num_db_tensors/hsize) # how many tensors to grab from db
mini_batch = 4 # batch size once tensors obtained from db and concatenated 
learning_rate = 0.001 # learning rate
nNeurons = 20 # number of neuronsining settings
tol = 1.0e0 # convergence tolerance on loss function


##### Instantiate the NN model and optimizer #####
model = NeuralNetwork(inputDim=ndIn, outputDim=ndOut, numNeurons=nNeurons).double()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


##### Training setup and variable initialization #####
istep = 0 # initialize the simulation step number to 0
iepoch = 1 # epoch number
torch_device = torch.device("cpu")
torch.set_num_threads(hsize)

# Average across hvd ranks
def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

# Training subroutine
def train(model, train_sampler, train_tensor_loader, epoch):
    #comm.Barrier()

    model.train()
    running_loss = 0.0
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    
    loss_fn = nn.functional.mse_loss

    for tensor_idx, tensor_keys in enumerate(train_tensor_loader):
        # grab data from database
        concat_tensor = torch.cat([torch.from_numpy(client.get_tensor(key)) \
                      for key in tensor_keys], dim=0)

        mbdata = MinibDataset(concat_tensor)
        train_loader = torch.utils.data.DataLoader(mbdata, shuffle=True, batch_size=mini_batch)
        for batch_idx, dbdata in enumerate(train_loader): 
            # split inputs and outputs
            target = dbdata[:, :ndOut]
            features = dbdata[:, ndOut:]

            optimizer.zero_grad()
            output = model.forward(features)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
    running_loss = running_loss / len(train_loader) / mini_batch
    loss_avg = metric_average(running_loss, 'running_loss')
    
    if hvd.rank() == 0: 
        print(f"Training set: Average loss: {loss_avg:>8e}")

    return features, loss_avg


##### While loop that checks when training data is available on database #####
if (hrank == 0):
    print("\nStarting training loop ... \n")
while True:
    # check to see if the time step number has been sent to database, if not cycle
    if (client.poll_tensor("step",0,1)):
        tmp = client.get_tensor('step')
    else:
        continue

    # new data is available in database so update it and train 1 epoch
    if (istep != tmp[0]): 
        istep = tmp[0]
        if (hrank == 0):
            print("\nGetting new training data from DB ...")
            print(f"Working on time step {istep} \n")

        datasetTrain = PhastaRankDataset(num_db_tensors,istep)
        # Horovod: use DistributedSampler to partition the training data
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                    datasetTrain, num_replicas=hsize, rank=hrank, drop_last=False)
        train_tensor_loader = torch.utils.data.DataLoader(
                    datasetTrain, batch_size=batch, sampler=train_sampler)
       
        if (iepoch==1):
            # Horovod: broadcast parameters & optimizer state.
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
            # Horovod: wrap optimizer with DistributedOptimizer.
            optimizer = hvd.DistributedOptimizer(optimizer,named_parameters=model.named_parameters(),
                                             op=Sum)
        if (hrank == 0):
            print(f"\n Epoch {iepoch}\n-------------------------------")
        
        features, global_loss = train(model, train_sampler, train_tensor_loader, iepoch)
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
        sleep(0.5)
        
    # new data is not avalable, so train another epoch on current data
    else:
        if (hrank == 0):
            print(f"\n Epoch {iepoch}\n-------------------------------")
        
        features, global_loss = train(model, train_sampler, train_tensor_loader, iepoch)
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
        sleep(0.5)

        

##### Save model to file before exiting #####
if (hrank == 0):
    model_name = "model"
    # usual way of saving model
    torch.save(model.state_dict(), f"{model_name}.pt", _use_new_zipfile_serialization=False)
    # save model to be used for online inference with SmartSim
    module = torch.jit.trace(model, features)
    torch.jit.save(module, f"{model_name}_jit.pt")
    print("")
    print("Saved model to disk\n")


##### Perform some predictions with the model
if (hrank == 0):
    print("Performing some predictions with the model \n")
    class CustomDataset(Dataset):
        def __init__(self, data): # initialize the class attributes
            self.data = data

        def __len__(self): # return number of samples in training data
            return len(self.data)

        def __getitem__(self,idx): # return sample from dataset at given index, splitting features and targets
            sample = self.data[idx]
            return sample

    inputs = np.random.uniform(low=0, high=10, size=(100,1))
    inputs_dataset = CustomDataset(inputs)
    inputs_dataloader = DataLoader(inputs_dataset, batch_size=100)
    model.eval()
    with torch.no_grad():
        for X in inputs_dataloader: 
            predictions = model(X).numpy()

    # plot results
    import matplotlib.pyplot as plt
    plt.plot(inputs, predictions, '.', label='Prediction')
    x = np.linspace(0, 10, 100)
    def f(x):
        return x**2 + 3*x + 1
    plt.plot(x, f(x), '-r', label='Target')
    plt.ylabel("f(x)")
    plt.xlabel("x")
    plt.legend(loc="upper left")
    plt.savefig('fig.pdf', bbox_inches='tight')


##### Exit and tell data loader to exit too
if (hrank == 0):
    print("Telling data loader to quit ... \n")
    arrMLrun = np.zeros(2)
    client.put_tensor("check-run",arrMLrun)

    print("Exiting ...")
    
 


