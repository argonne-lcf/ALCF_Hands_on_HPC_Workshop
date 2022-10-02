"""
configs.py

Implements various configuration objects.
"""
from __future__ import absolute_import, annotations, division, print_function
from collections.abc import Sequence
from dataclasses import dataclass, field
import datetime
import logging
import os
from pathlib import Path
import random
from typing import Any, Optional, Union
from typing import Callable

from ConfigSpace.api.distributions import Distribution
from hydra.core.config_store import ConfigStore
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data


log = logging.getLogger(__name__)


HERE = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = HERE.parent.parent
CONF_DIR = HERE.joinpath('conf')
LOGS_DIR = HERE.joinpath('logs')
AIM_DIR = HERE.joinpath('.aim')
OUTPUTS_DIR = HERE.joinpath('outputs')
DATA_DIR = HERE.joinpath('data')

CONF_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
OUTDIRS_FILE = OUTPUTS_DIR.joinpath('outdirs.log')


MODELS = {}


def add_to_outdirs_file(outdir: os.PathLike):
    with open(OUTDIRS_FILE, 'a') as f:
        f.write(Path(outdir).resolve().as_posix())
        f.write('\n')


def seed_everything(seed: int):
    pl.seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type:ignore
        torch.backends.cudnn.benchmark = False     # type:ignore
        torch.use_deterministic_algorithms(True)


def get_timestamp(fstr: Optional[str] = None):
    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)


@dataclass
class BaseConfig:
    def update(self, params: dict[str, Any]):
        for key, val in params.items():
            old_val = getattr(self, key, None)
            if old_val is not None:
                log.info(f'Updating {key} from: {old_val} to {val}')
                setattr(self, key, val)


@dataclass
class NetworkConfig:
    drop1: float
    drop2: float
    filters1: int
    filters2: int
    activation_fn: str
    hidden_size: int


@dataclass
class TrainerConfig:
    lr_init: float
    batch_size: int
    logfreq: int = 10
    epochs: int = 5
    seed: int = 9992
    num_threads: int = 16

    def scale_lr(self, factor: int) -> float:
        return self.lr_init * factor


@dataclass
class ExperimentConfig:
    trainer: TrainerConfig
    network: NetworkConfig
    wandb: Any


# @dataclass
# class NetworkConfig(BaseConfig):
#     filters: list[int]
#     sizes: list[int]
#     units: list[int]
#     out_features: int = 10
#     activation_fn: str = 'relu'
#     dropout_prob: float = 0.0
#     use_batch_norm: bool = True
#     pool: Optional[list[int]] = None
#
#     def __post_init__(self):
#         if self.pool is None:
#             self.pool = len(self.filters) * [2]
#         assert self.pool is not None
#
#     def update(self, params: dict[str, Any]):
#         for key, val in params.items():
#             old_val = getattr(self, key, None)
#             if old_val is not None:
#                 log.info(f'Updating {key} from: {old_val} to {val}')
#                 setattr(self, key, val)
#
#
# @dataclass
# class OptimizerConfig(BaseConfig):
#     name: str = 'Adam'
#     lr_init: float = 0.001
#
#     def __post_init__(self):
#         assert self.name.lower() in ['adam', 'sgd']


@dataclass
class lrSchedulerConfig(BaseConfig):
    scheduler: Any  # lr_scheduler
    # the unit of the scheduler's step size, could also be 'step'
    # 'epoch' updates the scheduler on epoch end whereas 'step'
    # updates it after an optimizer update
    interval: str = 'epoch'
    # How many epochs / steps should pass between calls to `scheduler.step()`.
    # 1 corresponds to updating the learning rate after every epoch / step
    frequency: Optional[int] = 1
    # metric to monitor for schedulers like `ReduceLROnPlateau`
    monitor: str = 'val_loss'
    # if set to `True`, will enforce that the value specified 'monitor'
    # is available when the scheduler is updated, thus stopping
    # training if not found.
    # If set to `False`, it will only produce a warning
    strict: bool = True


@dataclass
class BatchSizes(BaseConfig):
    train: int
    val: int
    test: int


Scalar = Union[float, int, bool, str]  # , np.ScalarType]
Hparam = Union[Scalar, str]


@dataclass
class HyperParameter:
    name: str
    value: list | tuple
    default_value: Optional[Hparam] = None


@dataclass
class hpInteger:
    name: str
    default: int
    bounds: tuple[int, int]
    log: bool = False
    distribution: Optional[Distribution] = None
    # distribution: Distribution = Uniform


@dataclass
class hpFloat:
    name: str
    default: float
    bounds: tuple[float, float]
    log: bool = False
    distribution: Optional[Distribution] = None


@dataclass
class hpCategorical:
    name: str
    items: Sequence[Union[str, int, float]]
    default: Optional[Union[str, int, float]] = None

    def as_hparam(self):
        hparam = {
            'name': self.name,
            'items': self.items,
        }
        if self.default is not None:
            hparam['default'] = self.default

        return hparam


@dataclass
class HyperParameters:
    # parameters: Optional[List[Dict]] = field(default_factory=list)
    # parameters: list[Any] = field(default_factory=list)
    parameters: dict[str, Scalar] = field(default_factory=dict)

    def __post_init__(self):
        self._space = {}
        # if self.parameters is not None:
        for name, param in self.parameters.items():
            self._space[name] = param
            # if str(param.type).lower() in ['int', 'integer']:
            #     self._space[name] = Integer(
            #         name=name,
            #         bounds=param.bounds,
            #         default=param.default,
            #         distribution=param.distribution,
            #         log=param.log,
            #     )
            # elif str(param.type).lower() == 'float':
            #     self._space[name] = Float(
            #         name=name,
            #         bounds=param.bounds,
            #         default=param.default,
            #         distribution=param.distribution,
            #         log=param.log,
            #     )
            # elif str(param.type).lower() == 'categorical':
            #     self._space[name] = Categorical(
            #         name=name,
            #         items=param.items,
            #         default=param.default,
            #     )
            # else:
            #     raise TypeError(f'Unexpected type for {name}')


            # import pdb; pdb.set_trace()
            # assert isinstance(param, dict)
            # assert isinstance(param, DictConfig)
            # keys = list(param.keys())
            # assert len(keys) > 0
            # name = keys[0]
            # hparam = HyperParameter(
            #     name=name,
            #     value=param.get('value', None),
            #     default_value=param.get('default_value', None),
            # )
            # self.params_dict[name] = hparam


# @dataclass
# class ExperimentConfig:
#     seed: int
#     batch_sizes: BatchSizes
#     network: NetworkConfig
#     optimizer: OptimizerConfig
#     xshape: List[int] = field(default_factory=list)
#     num_workers: int = 8
#     nranks: int = 1
#     ngpu_per_rank: int = 4
#     # num_workers: Optional[int] = field(default=8)
#     # num_workers: Optional[int] = 8
#     # xshape: Optional[List[int]] = field(default_factory=list)
#     name: Optional[str] = 'Experiment'
#     work_dir: Optional[str] = None
#     hparams: Optional[HyperParameters] = None
#     # run_id: Optional[Any] = None
#     extras: Optional[Any] = None
#     # hparams: Optional[dict[str, Any]] = None
#
#     def check_in_obj(self, key: str, obj: Any) -> bool:
#         return key in asdict(obj).keys()
#
#     def update(self, params: dict[str, Hparam]):
#         # for key, val in self.hparams._space.items():
#         self.network.update(params)
#         self.batch_sizes.update(params)
#         self.optimizer.update(params)
#         # if self.check_in_obj(key, self.batch_sizes):
#         #     self.batch_sizes.update()
#         # if key in asdict(self.batch_sizes).keys()


@dataclass
class DataObject:
    dataset: data.Dataset
    loader: data.DataLoader
    transform: Optional[
        list[Callable] | dict[str, Callable] | Callable
    ] = None


@dataclass
class ExperimentData:
    train: DataObject
    val: DataObject
    test: DataObject


def get_config(overrides: Optional[list[str]] = None):
    from hydra import (
        initialize_config_dir,
        compose
    )
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    overrides = [] if overrides is None else overrides
    with initialize_config_dir(
            CONF_DIR.absolute().as_posix(),
            # version_base=None,
    ):
        cfg = compose(
            'config',
            overrides=overrides,
            # return_hydra_config=True,
        )

    return cfg


# def get_experiment(
#         overrides: Optional[list[str]] = None,
#         build_networks: bool = True,
#         keep: Optional[str | list[str]] = None,
#         skip: Optional[str | list[str]] = None,
# ):
#     cfg = get_config(overrides)
#     if cfg.framework == 'pytorch':
#         # from l2hmc.experiment.pytorch.experiment import Experiment
#         return Experiment(
#             cfg,
#             keep=keep,
#             skip=skip,
#             build_networks=build_networks,
#         )
#     elif cfg.framework == 'tensorflow':
#         from l2hmc.experiment.tensorflow.experiment import Experiment
#         return Experiment(
#             cfg,
#             keep=keep,
#             skip=skip,
#             build_networks=build_networks,
#         )
#     else:
#         raise ValueError(
#             f'Unexpected value for `cfg.framework: {cfg.framework}'
#         )


cs = ConfigStore.instance()

cs.store(
    name='experiment_config',
    node=ExperimentConfig
)
