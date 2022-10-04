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
    hidden_size: int
    activation_fn: str = 'relu'


@dataclass
class DataConfig:
    batch_size: int = 128
    dataset: str = 'MNIST'


    def __post_init__(self):
        assert self.dataset in ['MNIST', 'FashionMNIST']


@dataclass
class TrainerConfig:
    lr_init: float
    logfreq: int = 10
    epochs: int = 5
    seed: int = 9992
    num_threads: int = 16
    # batch_size: int
    # dataset: str = 'MNIST'

    def scale_lr(self, factor: int) -> float:
        return self.lr_init * factor


@dataclass
class ExperimentConfig:
    data: DataConfig
    trainer: TrainerConfig
    network: NetworkConfig
    wandb: Any


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


cs = ConfigStore.instance()
cs.store(
    name='experiment_config',
    node=ExperimentConfig
)
