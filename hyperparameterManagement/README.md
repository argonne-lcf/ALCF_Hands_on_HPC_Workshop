<h1><span style="line-height:3.0em;font-size:1.5em;"> Hyperparameter Management <a href="https://hydra.cc"><img src="https://hydra.cc/img/logo.svg" width="10%" display="inline" style="vertical-align:middle;line-height:3.0em;margin-right:10%;" align="left" ></a> </span></h1>

**Author**: [Sam Foreman](https://samforeman.me) ([foremans@anl.gov](mailto:///foremans@anl.gov))

This section will cover some best practices / ideas related to experiment organization and hyperparameter management.

We use [Hydra](https://hydra.cc)[^1] for configuration management.

[^1]: [Hydra](https://hydra.cc): A framework for elegantly configuring complex applications


# Organization

```txt
📂 sdl_workshop/hyperparameterManagement/
┣━━ 📂 src/
┃   ┗━━ 📂 hplib/
┃       ┣━━ 📂 conf/
┃       ┃   ┣━━ 📂 network
┃       ┃   ┃   ┗━━ 📄 default.yaml
┃       ┃   ┣━━ 📂 sweeps
┃       ┃   ┃   ┗━━ 📄 default.yaml
┃       ┃   ┣━━ 📂 trainer
┃       ┃   ┃   ┗━━ 📄 default.yaml
┃       ┃   ┣━━ 📂 wandb
┃       ┃   ┃   ┗━━ 📄 default.yaml
┃       ┃   ┗━━ 📄 config.yaml
┃       ┣━━ 📂 utils/
┃       ┃   ┗━━ 🐍 pylogger.py
┃       ┣━━ 🐍 __init__.py
┃       ┣━━ 📄 affinity.sh
┃       ┣━━ 🐍 configs.py
┃       ┣━━ 🐍 main.py
┃       ┣━━ 🐍 network.py
┃       ┗━━ 🐍 trainer.py
┣━━ 📄 pyproject.toml
┗━━ 📄 setup.cfg
```
