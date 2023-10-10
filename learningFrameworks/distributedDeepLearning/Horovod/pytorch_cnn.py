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


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 16)')
parser.add_argument('--epochs', type=int, default=32, metavar='N',
                    help='number of epochs to train (default: 32)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'],
                    help='Whether this is running on cpu or gpu')
parser.add_argument('--num_threads', default=8, help='set number of threads per worker', type=int)
parser.add_argument('--num_workers', default=8, help='set number of io workers', type=int)
parser.add_argument('--wandb', action='store_true', 
                    help='whether to use wandb to log data')                
parser.add_argument('--project', default="sdl-pytorch-mnist", type=str)
parser.add_argument('--testing', action='store_true', default=False)
args = parser.parse_args()

t0 = time.time()

try:
    import wandb
    wandb.init(project=args.project)
    config = wandb.config          # Initialize config
    config.batch_size = args.batch_size         # input batch size for training (default: 64)
    config.test_batch_size = args.test_batch_size    # input batch size for testing (default: 1000)
    config.epochs = args.epochs            # number of epochs to train (default: 10)
    config.lr = args.lr              # learning rate (default: 0.01)
    config.momentum = args.momentum         # SGD momentum (default: 0.5) 
    config.device = args.device        # disables CUDA training
    config.seed = args.seed               # random seed (default: 42)
    config.log_interval = args.log_interval     # how many batches to wait before logging training status
    config.num_workers = args.num_workers
    
except:
    args.wandb = False

args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.device == 'gpu':
    torch.cuda.set_device(int(0))
    torch.cuda.manual_seed(args.seed)

if (args.num_threads!=0):
    torch.set_num_threads(args.num_threads)

print("Torch Thread setup: ")
print(" Number of threads: ", torch.get_num_threads())


kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.device == 'gpu' else {}
train_dataset = \
    datasets.MNIST('datasets/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,  **kwargs)

test_dataset = \
               datasets.MNIST('datasets', train=False, transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, **kwargs)
ntrain=len(train_loader.dataset)
ntest=len(test_loader.dataset)
print("Sample size: ", ntrain, ntest)
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


if args.device == 'gpu':
    # Move model to GPU.
    model.cuda()

#optimizer = optim.SGD(model.parameters(), lr=args.lr,
#                      momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    running_loss = torch.tensor(0.0)
    training_acc = torch.tensor(0.0)
    if args.device == "gpu":
        running_loss = running_loss.cuda()
        training_acc = training_acc.cuda()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.device == "gpu":
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        training_acc += pred.eq(target.data.view_as(pred)).float().sum()
        running_loss += loss

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( \
                    epoch, batch_idx * len(data), ntrain, 100. * batch_idx / len(train_loader), loss.item()/args.batch_size))
    running_loss /= ntrain
    training_acc /= ntrain
    print("Training set: Average loss: {:.4f}, Accuracy: {:.2f}%".format(running_loss, training_acc*100))
    return running_loss, training_acc


def test():
    model.eval()
    test_loss = torch.tensor(0.0)
    test_accuracy = torch.tensor(0.0)
    if args.device == "gpu":
        test_loss = test_loss.cuda()
        test_accuracy = test_accuracy.cuda()
    n = 0
    for data, target in test_loader:
        if args.device == "gpu":        
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        #test_loss += F.nll_loss(output, target, size_average=False).item()
        test_loss += F.nll_loss(output, target)#.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).float().sum()
        n=n+1

    # DDP: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= ntest
    test_accuracy /= ntest

    # Horovod: print output only on first rank.
    if rank == 0:
        print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))
    return test_loss, test_accuracy


epoch_times = []
for epoch in range(1, args.epochs + 1):
    e_start = time.time()
    training_loss, training_acc = train(epoch)
    if args.testing==1:
        test_loss, test_acc = test()
    e_end = time.time()
    epoch_times.append(e_end - e_start)
    print("Epoch - %d time: %s seconds" %(epoch, e_end - e_start))
    if (args.wandb):
        wandb.log({"time_per_epoch": e_end - e_start, 
            "train_loss": training_loss, "train_acc": training_acc, 
            "test_loss": test_loss, "test_acc":test_acc}, step=epoch)

t1 = time.time()
print("Total training time: %s seconds" %(t1 - t0))
print("Average time per epoch in the last 5: ", numpy.mean(epoch_times[-5:]))
