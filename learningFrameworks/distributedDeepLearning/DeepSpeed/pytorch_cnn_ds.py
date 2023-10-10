from __future__ import print_function
import os
import argparse
import time
import socket

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

#DeepSpeed: import module
import deepspeed

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--epochs', type=int, default=32, metavar='N',
                    help='number of epochs to train (default: 32)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--num_threads', default=8, help='set number of threads per worker', type=int)
parser.add_argument('--num_workers', default=8, help='set number of io workers', type=int)
parser.add_argument('--wandb', action='store_true', 
                    help='whether to use wandb to log data')                
parser.add_argument('--project', default="sdl-pytorch-mnist", type=str)
parser.add_argument('--testing', action='store_true', default=False)

# parser
parser = deepspeed.add_config_arguments(parser)

args = parser.parse_args()
# initialization
deepspeed.init_distributed()
t0 = time.time()

try:
    if (args.wandb):
        import wandb
        wandb.init(project=args.project)
        config = wandb.config          # Initialize config
        config.seed = args.seed               # random seed (default: 42)
        config.log_interval = args.log_interval     # how many batches to wait before logging training status
        config.num_workers = args.num_workers
    
except:
    args.wandb = False

args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)


if (args.num_threads!=0):
    torch.set_num_threads(args.num_threads)
if rank==0:
    print(" Number of workers: ", size)
    print(" Torch Thread setup: ")
    print(" Number of threads: ", torch.get_num_threads())


train_dataset = \
    datasets.MNIST('datasets/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_dataset = \
               datasets.MNIST('datasets', train=False, transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p=0.25, training=self.training)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()

parameters = filter(lambda p: p.requires_grad, model.parameters())

#DeepSpeed: initialize deep spead
model_engine, optimizer, train_loader, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=parameters, training_data=train_dataset)
__, __, test_loader, __ = deepspeed.initialize(
    args=args, model=model, training_data=test_dataset)


ntrain = len(train_loader.dataset)
ntest = len(test_loader.dataset)
torch.cuda.set_device(model_engine.local_rank)
torch.cuda.manual_seed(args.seed)

fp16 = model_engine.fp16_enabled()
if rank==0:
    print("Number of samples: ", ntrain, ntest)
    print(f'fp16={fp16}')
def train(epoch):
    model.train()
    running_loss = torch.tensor(0.0)
    training_acc = torch.tensor(0.0)
    running_loss = running_loss.to(model_engine.local_rank)
    training_acc = training_acc.to(model_engine.local_rank)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(model_engine.local_rank), target.to(
            model_engine.local_rank)            
        optimizer.zero_grad()
        if fp16:
            data=data.half()
        output = model_engine(data)
        loss = F.nll_loss(output, target)
        model_engine.backward(loss)
        model_engine.step()
        pred = output.data.max(1, keepdim=True)[1]
        training_acc += pred.eq(target.data.view_as(pred)).float().sum()
        running_loss += loss

        if batch_idx % args.log_interval == 0 and rank==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( \
                                                                            epoch, batch_idx * len(data), ntrain, 100. * batch_idx / len(train_loader), loss.item()))
    running_loss /= ntrain/size
    training_acc /= ntrain/size
    if rank==0: print("Training set: Average loss: {:.4f}, Accuracy: {:.2f}%".format(running_loss, training_acc*100))
    return running_loss, training_acc


def test():
    model.eval()
    test_loss = torch.tensor(0.0)
    test_accuracy = torch.tensor(0.0)
    test_loss = test_loss.cuda()
    test_accuracy = test_accuracy.cuda()
    n = 0
    test_loss = test_loss.to(model_engine.local_rank)
    test_accuracy = test_accuracy.to(model_engine.local_rank)

    for data, target in test_loader:
        data, target = data.to(model_engine.local_rank), target.to(
            model_engine.local_rank)
        if fp16:
            data=data.half()
        output = model_engine(data)
        # sum up batch loss
        #test_loss += F.nll_loss(output, target, size_average=False).item()
        test_loss += F.nll_loss(output, target)#.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).float().sum()
        n=n+1

    # DDP: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= ntest/size
    test_accuracy /= ntest/size

    # Horovod: print output only on first rank.
    if rank == 0:
        print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))
    return test_loss, test_accuracy


epoch_times = []
for epoch in range(1, args.epochs + 1):
    e_start = time.time()
    training_loss, training_acc = train(epoch)
    if args.testing:
        test_loss, test_acc = test()
    e_end = time.time()
    epoch_times.append(e_end - e_start)
    if rank==0: print("Epoch - %d time: %s seconds" %(epoch, e_end - e_start))
    if (args.wandb):
        wandb.log({"time_per_epoch": e_end - e_start, 
            "train_loss": training_loss, "train_acc": training_acc, 
            "test_loss": test_loss, "test_acc":test_acc}, step=epoch)

t1 = time.time()
if rank==0:
    print("Total training time: %s seconds" %(t1 - t0))
    print("Average time per epoch in the last 5: ", numpy.mean(epoch_times[-5:]))
