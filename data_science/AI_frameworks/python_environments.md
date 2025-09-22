# Python Environments

**Learning Goals:**

* How to add prebuilt Python environments into your environment
* Create a custom build environment based on a pre-existing Conda module


## Overview

ALCF provides a pre-built Anaconda environment that makes available [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), [Horovod](https://horovod.readthedocs.io/en/stable/tensorflow.html), and [mpi4py](https://mpi4py.readthedocs.io/en/stable/) with Intel extensions and optimizations, among other popular Python and ML packages.
This Anaconda environment can be activated loading the `frameworks` module.


## The AI/ML `frameworks` module

The following command can be used to load the latest [`frameworks` module](https://docs.alcf.anl.gov/aurora/data-science/python/#aiml-framework-module)
```bash
module load frameworks
```

Please note that:

- The `frameworks` module automatically activates a pre-built `conda` environment which comes with GPU-supported builds of Python AI/ML libraries (use `pip list` to see the list of all installed packages).
- The `frameworks` module may load a different oneAPI compiler SDK than the default module.
- The `frameworks` module is updated approximately every quarter.

If you need to install additional packages, there are two approaches covered in the following sections:

1. [Virtual environments via `venv`](#virtual-environment-via-venv): builds an extendable enviroment on top of the immutable `frameworks` environment.
2. [Clone the `frameworks` environment](#clone-conda-environment): complete mutable copy of the `frameworks` environment into a user's space.

> ⚠️  **Note**: Importing Python modules at large node counts (beyond 1000 nodes) from a _user-created_ virtual or conda environment can be significantly slow, or it may even crash the Lustre file system. Please refer to [ai_at_scale.md](ai_at_scale.md) on how to efficiently load custom-installed Python packages at scale using [Copper](https://docs.alcf.anl.gov/aurora/data-management/copper/copper/).


## Virtual Environment via `venv`

The easiest method for making a custom environment that builds on-top of the `frameworks` environment is to use `venv`. 
```bash
module load frameworks
python -m venv /path/to/venv --system-site-packages
```

By passing the `--system-site-packages` flag, the new virtual environment inherits all the packages from the `frameworks` environment, while being able to install new packages.

To activate this new environment,
```bash
source /path/to/venv/bin/activate
```

Once activated, installing packages with pip is as usual:
```bash
python -m pip install <new-package>
```

To install a _different version_ of a package that is **already installed** in the
`frameworks` environment add the `--ignore-installed` to your command:
```bash
python -m pip install --ignore-installed <new-package>
```

The base environment is not writable, so it is not possible to remove or uninstall packages from it. The packages installed with the above pip command should shadow those installed in the base environment.


## Clone Conda Environment

> ⚠️  **Note**: This approach takes several GB of disk space and is quite slow to complete. 
For these reasons, we suggest the use of [Python virtual environments](#virtual-environment-via-venv) whenever possible.

Cloning a Conda Environment creates a full copy of the ALCF conda environment in a specified directory. This means the user has full control of the environment. 

Create a `clone` of the `frameworks` environment by:

```bash
# load the frameworks module
module load frameworks
# create the clone
conda create --clone $CONDA_PREFIX --prefix /path/to/envs/myclone
# load the cloned environment
conda activate /path/to/envs/myclone
```

Future loading can be done with:
```bash
module load frameworks && conda activate /path/to/envs/myclone
```
It is necessary to ensure the same version of conda you are loading is the same as that with which you generated the clone.



## Additional Resources

- [ALCF Docs: Python on Aurora](https://docs.alcf.anl.gov/aurora/data-science/python/)

## [NEXT: -> PyTorch](pytorch_ddp.md)
