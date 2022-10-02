"""
ddp/network.py
"""
from __future__ import absolute_import, annotations, division, print_function
import torch

import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig

from hplib.configs import NetworkConfig


ACTIVATION_FNS = {
    'elu': nn.ELU(),
    'gelu': nn.GELU(),
    'tanh': nn.Tanh(),
    'relu': nn.ReLU(),
    'swish': nn.SiLU(),
    'leaky_relu': nn.LeakyReLU()
}


class Net(nn.Module):
    def __init__(
            self,
            config: dict | NetworkConfig | DictConfig,
    ):
        super(Net, self).__init__()
        self._with_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self._with_cuda else 'cpu'
        if isinstance(config, (dict, DictConfig)):
            self.config = instantiate(config)
        elif isinstance(config, NetworkConfig):
            self.config = config
        assert isinstance(self.config, NetworkConfig)
        self.activation_fn = ACTIVATION_FNS.get(
            self.config.activation_fn.lower(),
            None
        )
        assert callable(self.activation_fn)
        self.layers = nn.ModuleList()
        self.layers.extend([
            nn.LazyConv2d(self.config.filters1, 3),
            nn.LazyConv2d(self.config.filters2, 3),
            nn.MaxPool2d(2),
        ])
        self.layers.append(self.activation_fn)
        self.layers.extend([
            nn.Flatten(),
            nn.Dropout(self.config.drop1),
            nn.LazyLinear(self.config.hidden_size),
            nn.Dropout(self.config.drop2),
            nn.LazyLinear(10)
        ])

        if torch.cuda.is_available():
            self.cuda()
            self.layers.cuda()

        # self.conv1 = nn.Conv2d(
        #     1,
        #     self.config.filters1,
        #     3,
        #     1
        # )
        # self.conv2 = nn.Conv2d(
        #     self.config.filters1,
        #     self.config.filters2,
        #     3,
        #     1
        # )
        # self.dropout1 = nn.Dropout(self.config.drop1)
        # self.dropout2 = nn.Dropout(self.config.drop2)
        # self.fc1 = nn.Linear(9216, self.config.hidden_size)
        # self.fc2 = nn.Linear(self.config.hidden_size, 10)

    def get_config(self):
        return self.config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        # x = F.relu(self.fc1(torch.flatten(x, 1)))
        # x = self.dropout2(x)
        # x = self.fc2(x)
        x.requires_grad_(True)
        x = x.to(self.device)
        for layer in self.layers:
            x = layer(x)

        return F.log_softmax(x, dim=1)




def get_network(config: NetworkConfig) -> nn.Module:
    return Net(config)
