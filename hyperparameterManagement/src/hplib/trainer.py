"""
ddp/trainer.py
"""
from __future__ import absolute_import, annotations, division, print_function
import time
from typing import Optional

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from torch import optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.grad_scaler import GradScaler
import torch.distributed as dist
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms

from hplib.network import Net
from hplib.configs import DATA_DIR, TrainerConfig, NetworkConfig
from hplib.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def metric_average(val: torch.Tensor, size: int = 1):
    try:
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
    except Exception as exc:
        log.exception(exc)
        pass

    # if (WITH_DDP):
    #     # Sum everything and divide by the total size
    #     dist.all_reduce(val, op=dist.ReduceOp.SUM)
    #     return val / SIZE

    # return val
    return val / size


import wandb

class Trainer:
    def __init__(
            self,
            config: TrainerConfig | dict | DictConfig,
            net_config: Optional[NetworkConfig | dict | DictConfig],
            wbrun: Optional[wandb.run] = None,
            scaler: Optional[GradScaler] = None,
            model: Optional[torch.nn.Module] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        if isinstance(config, (dict, DictConfig)):
            self.config = instantiate(config)
        elif isinstance(config, TrainerConfig):
            self.config = config
        if not isinstance(config, TrainerConfig):
            import pdb; pdb.set_trace()
        # self.cfg = cfg
        # self.rank = RANK
        if scaler is None:
            self.scaler = None

        if net_config is None:
            self.net_config = model.config
        else:
            if isinstance(net_config, (dict, DictConfig)):
                self.net_config = instantiate(net_config)
            else:
                self.net_config = net_config

        assert isinstance(self.config, TrainerConfig)
        assert isinstance(self.net_config, NetworkConfig)

        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.loss_fn = nn.CrossEntropyLoss()
        self._ngpus = 1
        self.world_size = 1
        self.rank = 0
        # self.setup_torch()
        self.data = self.setup_data()
        if model is None:
            self.model = self.build_model(self.net_config)
        if optimizer is None:
            self.optimizer = self.build_optimizer(
                model=self.model,
                lr_init=self.config.lr_init
            )

        if torch.cuda.is_available():
            self.model.cuda()
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_size = torch.cuda.device_count()
            self._ngpus = self.world_size * torch.cuda.device_count()
            if self._ngpus > 1:
                self.model = DDP(self.model)

        self.wbrun = wbrun
        # if self.rank == 0:
        # if self.wbrun is wandb.run:
        # if isinstance(self.wbrun, wandb.run):
        # if self.wbrun is None:
        #     log.warning(f'Initializing WandB from {self.rank}')
        #     # self.wbrun = wandb.init()
        # assert self.wbrun is wandb.run
        # self.optimizer = self.build_optimizer(self.model)
        # if WITH_CUDA:
        #    self.loss_fn = self.loss_fn.cuda()
        # self.backend = self.cfg.backend
        # if WITH_DDP:
        # init_process_group(RANK, SIZE, backend='DDP')
        if self.wbrun is not None:
            log.warning(f'Caught wandb.run from: {self.rank}')
            self.wbrun.watch(self.model, log='all', criterion=self.loss_fn)


    def build_model(
            self,
            net_config: Optional[NetworkConfig] = None
    ) -> nn.Module:
        if net_config is None:
            net_config = self.config.network

        assert net_config is not None
        model = Net(net_config)
        xshape = (1, *[28, 28])
        x = torch.rand((self.config.batch_size, *xshape))
        if torch.cuda.is_available():
            model.cuda()
            x = x.cuda()

        _ = model(x)

        return model

    def build_optimizer(
            self,
            lr_init: float,
            model: nn.Module
    ) -> torch.optim.Optimizer:
        return optim.Adam(model.parameters(), lr=lr_init)

    def setup_torch(self):
        torch.manual_seed(self.config.seed)
        # if self.device == 'gpu':
        if torch.cuda.is_available():
            # DDP: pin GPU to local rank
            # torch.cuda.set_device(int(LOCAL_RANK))
            torch.cuda.manual_seed(self.config.seed)

        if (
                self.config.num_threads is not None
                and isinstance(self.config.num_threads, int)
                and self.config.num_threads > 0
        ):
            torch.set_num_threads(self.config.num_threads)

            log.info('\n'.join([
                'Torch Thread Setup:',
                f' Number of threads: {torch.get_num_threads()}',
            ]))

    def setup_data(self):
        kwargs = {}

        if self.device == 'gpu':
            kwargs = {'num_workers': 0, 'pin_memory': True}

        train_dataset = (
            datasets.MNIST(
                DATA_DIR,
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ])
            )
        )

        # DDP: use DistributedSampler to partition training data
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=self.world_size, rank=self.rank,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            **kwargs
        )

        test_dataset = (
            datasets.MNIST(
                DATA_DIR,
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            )
        )
        # DDP: use DistributedSampler to partition the test data
        test_sampler = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=self.world_size, rank=self.rank
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config.batch_size
        )

        return {
            'train': {
                'sampler': train_sampler,
                'loader': train_loader,
            },
            'test': {
                'sampler': test_sampler,
                'loader': test_loader,
            }
        }

    def train_step(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        self.optimizer.zero_grad()
        probs = self.model(data)
        loss = self.loss_fn(probs, target)

        if self.scaler is not None and isinstance(self.scaler, GradScaler):
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        _, pred = probs.data.max(1)
        acc = (pred == target).sum()

        return loss, acc

    def train_epoch(
            self,
            epoch: int,
    ) -> dict:
        self.model.train()
        # start = time.time()
        running_acc = torch.tensor(0.)
        running_loss = torch.tensor(0.)
        if torch.cuda.is_available():
            running_acc = running_acc.cuda()
            running_loss = running_loss.cuda()

        train_sampler = self.data['train']['sampler']
        train_loader = self.data['train']['loader']
        # DDP: set epoch to sampler for shuffling
        train_sampler.set_epoch(epoch)
        for bidx, (data, target) in enumerate(train_loader):
            t0 = time.time()
            loss, acc = self.train_step(data, target)
            running_acc += acc
            running_loss += loss.item()
            if bidx % self.config.logfreq == 0 and self.rank == 0:
                # DDP: use train_sampler to determine the number of
                # examples in this workers partition
                metrics = {
                    'epoch': epoch,
                    'dt': time.time() - t0,
                    'batch_acc': acc.item() / self.config.batch_size,
                    'batch_loss': loss.item() / self.config.batch_size,
                    'acc': running_acc / len(self.data['train']['sampler']),
                    'running_loss': (
                        running_loss / len(self.data['train']['sampler'])
                    ),
                }
                pre = [
                    f'[{self.rank}]',
                    (   # looks like: [num_processed/total (% complete)]
                        f'[{epoch}/{self.config.epochs}:'
                        f' {bidx * len(data)}/{len(train_sampler)}'
                        f' ({100. * bidx / len(train_loader):.0f}%)]'
                    ),
                ]
                log.info(' '.join([
                    *pre, *[f'{k}={v:.4f}' for k, v in metrics.items()]
                ]))
                if self.rank == 0:  # and self.wbrun is wandb.run:
                    self.wbrun.log(
                        {f'batch/{key}': val for key, val in metrics.items()}
                    )


        running_loss = running_loss / len(train_sampler)
        running_acc = running_acc / len(train_sampler)
        training_acc = metric_average(running_acc, size=self._ngpus)
        loss_avg = metric_average(running_loss, size=self._ngpus)
        if self.rank == 0:
            self.wbrun.log({'train/loss': loss_avg, 'train/acc': training_acc})

        return {'loss': loss_avg, 'acc': training_acc}

    def test(self) -> float:
        total = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for data, target in self.data['test']['loader']:
                if self.device == 'gpu':
                    data, target = data.cuda(), target.cuda()

                probs = self.model(data)
                _, predicted = probs.data.max(1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        acc = correct / total
        if self.rank == 0:
            self.wbrun.log({'test/acc': correct / total})

        return acc
