"""
ddp/trainer.py
"""
from __future__ import absolute_import, annotations, division, print_function
import time
from typing import Optional, Any

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from torch import optim
from torch.cuda.amp.grad_scaler import GradScaler
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
import wandb

from hplib.configs import DATA_DIR, ExperimentConfig, NetworkConfig
from hplib.network import Net
from hplib.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def metric_average(val: torch.Tensor, size: int = 1):
    try:
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
    except Exception as exc:
        log.exception(exc)
        pass

    return val / size


class Trainer:
    def __init__(
            self,
            config: ExperimentConfig | dict | DictConfig,
            wbrun: Optional[Any] = None,
            scaler: Optional[GradScaler] = None,
            model: Optional[torch.nn.Module] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        # if isinstance(config, (dict, DictConfig)):
        #     self.config = instantiate(config)
        # elif isinstance(config, TrainerConfig):
        #     self.config = config
        # assert isinstance(self.config, TrainerConfig)
        if isinstance(config, (dict, DictConfig)):
            self.config = instantiate(config)
        elif isinstance(config, ExperimentConfig):
            self.config = config
        else:
            raise TypeError(
                'Expected `config` to be of type: '
                '`dict | DictConfig | ExperimentConfig`'
            )

        if scaler is None:
            self.scaler = None

        # if net_config is None:
        #     assert model is not None and hasattr(model, 'config')
        #     self.net_config = model.config
        # else:
        #     if isinstance(net_config, (dict, DictConfig)):
        #         self.net_config = instantiate(net_config)
        #     else:
        #         self.net_config = net_config

        # assert isinstance(self.config.train, TrainerConfig)
        # assert isinstance(self.net_config, NetworkConfig)
        assert isinstance(self.config, ExperimentConfig)

        self.loss_fn = nn.CrossEntropyLoss()
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'

        self.rank = 0
        self._ngpus = 1
        self.world_size = 1
        # self.setup_torch()
        self.data = self.setup_data()
        if model is None:
            self.model = self.build_model(self.config.network)
        if optimizer is None:
            self.optimizer = self.build_optimizer(
                model=self.model,
                lr_init=self.config.trainer.lr_init
            )

        if torch.cuda.is_available():
            self.model.cuda()
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_size = torch.cuda.device_count()
            self._ngpus = self.world_size * torch.cuda.device_count()

        self.wbrun = wbrun
        if self.wbrun is not None:
            log.warning(f'Caught wandb.run from: {self.rank}')
            self.wbrun.watch(self.model, log='all', criterion=self.loss_fn)

    def build_model(self, net_config: NetworkConfig) -> nn.Module:
        assert net_config is not None
        model = Net(net_config)
        xshape = (1, *self._xshape)
        x = torch.rand((self.config.data.batch_size, *xshape))
        if torch.cuda.is_available():
            model.cuda()
            x = x.cuda()

        _ = model(x)

        if self.world_size > 1:
            model = DDP(model)

        return model

    def build_optimizer(
            self,
            lr_init: float,
            model: nn.Module
    ) -> torch.optim.Optimizer:
        return optim.Adam(model.parameters(), lr=lr_init)

    def setup_torch(self):
        torch.manual_seed(self.config.trainer.seed)
        # if self.device == 'gpu':
        if torch.cuda.is_available():
            # DDP: pin GPU to local rank
            # torch.cuda.set_device(int(LOCAL_RANK))
            torch.cuda.manual_seed(self.config.trainer.seed)

        if (
                self.config.trainer.num_threads is not None
                and isinstance(self.config.trainer.num_threads, int)
                and self.config.trainer.num_threads > 0
        ):
            torch.set_num_threads(self.config.trainer.num_threads)

            log.info('\n'.join([
                'Torch Thread Setup:',
                f' Number of threads: {torch.get_num_threads()}',
            ]))

    def get_mnist_datasets(self)-> dict[str, torch.utils.data.Dataset]:
        train_dataset = (
            datasets.MNIST(
                DATA_DIR.as_posix(),
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ])
            )
        )
        test_dataset = (
            datasets.MNIST(
                DATA_DIR.as_posix(),
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            )
        )
        self._xshape = [28, 28]
        return {
            'train': train_dataset,
            'test': test_dataset,
        }


    def get_fashionmnist_datasets(self)-> dict[str, torch.utils.data.Dataset]:
        train_dataset = (
            datasets.FashionMNIST(
                DATA_DIR.as_posix(),
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ])
            )
        )
        test_dataset = (
            datasets.FashionMNIST(
                DATA_DIR.as_posix(),
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            )
        )
        self._xshape = [28, 28]
        return {
            'train': train_dataset,
            'test': test_dataset,
        }


    def get_datasets(self, dset: str) -> dict[str, torch.utils.data.Dataset]:
        assert dset in datasets.__all__
        dset_obj = __import__(f'{datasets}.{dset}')
        assert isinstance(dset_obj, torch.utils.data.Dataset)
        train_dataset = (
            dset_obj(
                DATA_DIR.as_posix(),
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                ])
            )
        )
        test_dataset = (
            dset_obj(
                DATA_DIR.as_posix(),
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                ])
            )
        )
        return {
            'train': train_dataset,
            'test': test_dataset,
        }


    def setup_data(
            self,
            datasets: Optional[dict[str, torch.utils.data.Dataset]] = None,
    ):
        kwargs = {}

        if self.device == 'gpu':
            kwargs = {'num_workers': 0, 'pin_memory': True}

        if datasets is None:
            # datasets = self.get_mnist_datasets()
            # if self.config.dataset.lower() == 'fashionmnist': 
            if self.config.data.dataset.lower() == 'fashionmnist':
                datasets = self.get_fashionmnist_datasets()
            else:
                datasets = self.get_mnist_datasets()

        assert 'train' in datasets and 'test' in datasets
        train_dataset = datasets['train']
        test_dataset = datasets['test']

        self._xshape = [28, 28]
        # DDP: use DistributedSampler to partition training data
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=self.world_size, rank=self.rank,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.data.batch_size,
            sampler=train_sampler,
            **kwargs
        )

        # DDP: use DistributedSampler to partition the test data
        test_sampler = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=self.world_size, rank=self.rank
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config.data.batch_size
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

    def has_wbrun(self):
        assert self.wbrun is not None
        return (
            self.rank == 0
            and self.wbrun is not None
            and self.wbrun is wandb.run
        )

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
            if bidx % self.config.trainer.logfreq == 0 and self.rank == 0:
                # DDP: use train_sampler to determine the number of
                # examples in this workers partition
                metrics = {
                    'epoch': epoch,
                    'dt': time.time() - t0,
                    'batch_acc': acc.item() / self.config.data.batch_size,
                    'batch_loss': loss.item() / self.config.data.batch_size,
                    'acc': running_acc / len(self.data['train']['sampler']),
                    'running_loss': (
                        running_loss / len(self.data['train']['sampler'])
                    ),
                }
                pre = [
                    f'[{self.rank}]',
                    (   # looks like: [num_processed/total (% complete)]
                        f'[{epoch}/{self.config.trainer.epochs}:'
                        f' {bidx * len(data)}/{len(train_sampler)}'
                        f' ({100. * bidx / len(train_loader):.0f}%)]'
                    ),
                ]
                log.info(' '.join([
                    *pre, *[f'{k}={v:.4f}' for k, v in metrics.items()]
                ]))
                # if (
                #         self.rank == 0
                #         and self.wbrun is not None
                #         and self.wbrun is wandb.run
                # ):
                # if self.has_wbrun():
                # assert self.wbrun is not None
                # assert self.wbrun is not None and self.wbrun is wandb.run
                try:
                    self.wbrun.log(  # type:ignore
                        {f'batch/{key}': val for key, val in metrics.items()}
                    )
                except Exception:
                    pass

        running_loss = running_loss / len(train_sampler)
        running_acc = running_acc / len(train_sampler)
        training_acc = metric_average(running_acc, size=self._ngpus)
        loss_avg = metric_average(running_loss, size=self._ngpus)
        if self.rank == 0:
            assert (
                self.wbrun is not None 
                and self.wbrun is wandb.run  # type:ignore
            )
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
            try:
                self.wbrun.log({'test/acc': correct / total})  # type:ignore
            except Exception:
                pass

        return acc
