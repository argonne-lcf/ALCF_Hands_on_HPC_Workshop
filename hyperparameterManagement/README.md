<h1><span style="line-height:3.0em;font-size:1.5em;"> Hyperparameter Management <a href="https://hydra.cc"><img src="https://hydra.cc/img/logo.svg" width="10%" display="inline" style="vertical-align:middle;line-height:3.0em;margin-right:10%;" align="left" ></a> </span></h1>

**Author**: [Sam Foreman](https://samforeman.me) ([foremans@anl.gov](mailto:///foremans@anl.gov))

This section will cover some best practices / ideas related to experiment organization and hyperparameter management.

The accompanying slides for this talk can be found at: 📊 [Hyperparameter Management](https://saforem2.github.io/hparam-management-sdl2022/#/)

We use [Hydra](https://hydra.cc)[^1] for configuration management.

[^1]: [Hydra](https://hydra.cc): A framework for elegantly configuring complex applications

# Organization

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

## Quick Start

1. Clone the github repo and navigate into this directory
  ```shell
  $ git clone https://github.com/argonne-lcf/sdl_workshop
  $ cd sdl_workshop/hyperparameterManagement/
  ```
  
2. Create a virtualenv and perform local install
  ```shell
  $ python3 -m venv venv
  $ source venv/bin/activate
  $ python3 -c "import hplib; print(hplib.__file__)"
  ```

3. Run experiments:
  ```shell
  $ cd src/hplib
  $ ./train.sh
  ```
  
This will perform a complete training + evaluation run using the default configurations specified in [`src/hplib/conf/config.yaml`](./src/hplib/conf/config.yaml)
