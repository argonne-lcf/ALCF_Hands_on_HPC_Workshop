<h1><span style="line-height:3.0em;font-size:1.5em;"> Hyperparameter Management <a href="https://hydra.cc"><img src="https://hydra.cc/img/logo.svg" width="10%" display="inline" style="vertical-align:middle;line-height:3.0em;margin-right:10%;" align="left" ></a> </span></h1>

**Author**: [Sam Foreman](https://samforeman.me) ([foremans@anl.gov](mailto:///foremans@anl.gov))

This section will cover some best practices / ideas related to experiment organization and hyperparameter management.

We use [Hydra](https://hydra.cc) for configuration management.

- The slides for this talk can be found at: 
  - 📊 [Hyperparameter Management](https://saforem2.github.io/hparam-management-sdl2022/#/)

- There is a WandB workspace available online at:
  - [wandb.ai/alcf-mlops/sdl-wandb](https://wandb.ai/alcf-mlops/sdl-wandb?workspace=user-foremans)


## Contents
- [Quick Start](#quick-start)
  * [Modifying the Configuration](#modifying-the-configuration)
  * [Override Default Options](#override-default-options)
- [WandB Sweeps](#wandb-sweeps)
- [Project Organization](#project-organization)


# Quick Start

1. Clone the GitHub repo and navigate into this directory
  ```shell
  $ git clone https://github.com/argonne-lcf/sdl_workshop
  $ cd sdl_workshop/hyperparameterManagement/
  ```
2. Create a virtualenv and perform local install
  ```shell
  $ python3 -m venv venv
  $ source venv/bin/activate
  $ python3 -m pip install --upgrade pip
  $ python3 -m pip install -e .
  $ python3 -c "import hplib; print(hplib.__file__)"
  ```
3. Run experiments:
  ```shell
  $ cd src/hplib
  $ ./train.sh > train.log 2>&1 &
  $ tail -f train.log $(tail -1 logs/latest)
  ```
  
This will perform a complete training + evaluation run using the default configurations specified in [`src/hplib/conf/config.yaml`](./src/hplib/conf/config.yaml)

## Modifying the Configuration

Our default config looks like:

```yaml
# @package _global_
_target_: hplib.configs.ExperimentConfig

# Specify here default configuration
# Ordering determines precedence, i.e.
# the order determines the order in which options are overridden
defaults:
  - _self_
  - network: default.yaml
  - data: default.yaml
  - trainer: default.yaml
  - wandb: default.yaml
  # Nicely formatted / colored logs
  - override hydra/hydra_logging: colorlog
  - override hydra/job_logging: colorlog

hydra:
  job:
    chdir: true
```

The `_target_` field indicates that this configuration implements a `hplib.configs.ExperimentConfig` object.

Hydra will step recursively into nested fields and look for the implementation details specified by their values.

Explicitly, we have the following object implemented in [`src/hplib/configs.py`](./src/hplib/configs.py):

```python
from dataclasses import dataclass

 @dataclass
 class DataConfig:
     batch_size: int = 128
     dataset: str = 'MNIST'

 @dataclass
 class NetworkConfig:
     drop1: float
     drop2: float
     filters1: int
     filters2: int
     hidden_size: int
     activation_fn: str = 'relu'

@dataclass
class TrainerConfig:
    lr_init: float
    logfreq: int = 10
    epochs: int = 5
    seed: int = 9992
    num_threads: int = 16

    def scale_lr(self, factor: int) -> float:
        return self.lr_init * factor
     
@dataclass
class ExperimentConfig:
    data: DataConfig
    trainer: TrainerConfig
    network: NetworkConfig
    wandb: Any
```

We can use `hydra.utils.instantiate` to automatically instantiate an `ExperimentConfig` as follows:

```python
from hydra.utils import instantiate
from hplib.configs import ExperimentConfig

@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    config = instantiate(cfg)
    assert isinstance(config, ExperimentConfig)
    assert isinstance(config.trainer, TrainerConfig)
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.network, NetworkConfig)
```

## Override Default Options

We can override the default options in our configuration either:

1. By manually editing the `conf/network/default.yaml`
2. Specifying new values directly from the CLI:
  ```shell
  $ ./train.sh network.hidden_size=32 data.batch_size=2048
  ```
  will set the hidden size to be 32 in our `NetworkConfig`, and the batch size to be `2048` in our `DataConfig`
  
  
# WandB Sweeps

We define the set of hyperparameters we wish to optimize in the [`src/hplib/conf/sweeps/*.yaml`](./src/hplib/conf/sweeps/) files.

```yaml
name: min-loss-distributed
 method: bayes
 description: Find hparams which minimize batch/batch_loss

 metric:
   name: batch/batch_loss
   goal: minimize

 parameters:
   # -- TRAINER ----
   trainer.lr_init:
     values: [0.00001, 0.001, 0.01]
   # -- FIXED ------
   trainer.epochs:
     value: 2
   # -- NETWORK --------------
   network.drop1:
       values: [0.1, 0.2, 0.5]
   network.drop2:
       values: [0.1, 0.2, 0.5]

# -------------------------------------------------------------
# NOTE: launch INDIVIDUAL (distributed) agents, SEQUENTIALLY
program: './train.sh'  # run distributed training w/ DDP
# -------------------------------------------------------------

command:
  - ${env}
  - ${program}
  - ${args_no_hyphens}
```

# Project Organization

```txt
📂 sdl_workshop/hyperparameterManagement/
┣━━ 📂 src/
┃   ┗━━ 📂 hplib/
┃       ┣━━ 📂 conf/
┃       ┃   ┣━━ 📂 data/
┃       ┃   ┃   ┗━━ 📄 default.yaml
┃       ┃   ┣━━ 📂 network/
┃       ┃   ┃   ┗━━ 📄 default.yaml
┃       ┃   ┣━━ 📂 sweeps/
┃       ┃   ┃   ┣━━ 📄 max-acc-distr.yaml
┃       ┃   ┃   ┣━━ 📄 max-acc-single.yaml
┃       ┃   ┃   ┣━━ 📄 min-loss-distr.yaml
┃       ┃   ┃   ┗━━ 📄 min-loss-single.yaml
┃       ┃   ┣━━ 📂 trainer/
┃       ┃   ┃   ┗━━ 📄 default.yaml
┃       ┃   ┣━━ 📂 wandb/
┃       ┃   ┃   ┗━━ 📄 default.yaml
┃       ┃   ┗━━ 📄 config.yaml
┃       ┣━━ 📂 utils/
┃       ┃   ┗━━ 🐍 pylogger.py
┃       ┣━━ 🐍 __init__.py
┃       ┣━━ 📄 affinity.sh
┃       ┣━━ 🐍 configs.py
┃       ┣━━ 🐍 main.py
┃       ┣━━ 🐍 network.py
┃       ┣━━ 📄 train.sh
┃       ┗━━ 🐍 trainer.py
┣━━ 📄 pyproject.toml
┣━━ 📄 README.md
┗━━ 📄 setup.cfg
```
