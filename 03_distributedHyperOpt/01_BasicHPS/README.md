# Hyperparameter Search for Deep Learning (Basic)

Every DeepHyper search requires at least 2 Python objects as input:

- `run`: Your "black-box" function returning the objective value to be maximized
- `Problem`: an instance of `deephyper.problem.BaseProblem` which defines the search space of input parameters to `run`.

We will illustrate DeepHyper HPS for the MNIST dataset, with a goal of tuning the hyperparameters to maximize the classification accuracy.

## Environment setup on ALCF's`theta`:

To start on Theta, let's set up  a clean workspace and download the Keras benchmark model and MNIST data.

```bash
# Create a new workspace with Balsam DB
module unload balsam  # Unload Balsam module: we want to use DeepHyper which comes with everything
module load deephyper/0.2.1  # Includes Balsam, TensorFlow, Keras, etc...
rm -r ~/.balsam  # reset default settings (for now)
mkdir ~/dh-tutorial
cd ~/dh-tutorial
balsam init db
source balsamactivate db
```



## Setting up the problem

**TODO**: 

- Modify paths below to be consistent with Theta environment setup from above

Start by creating a new DeepHyper project workspace. This is a directory where you will create search problem instances that are automatically installed and importable across your Python environment.

```bash
export DISABLE_PYMODULE_LOG=True
deephyper start-project hps_demo
```

A new `hps_demo` directory is created, containing the following files:

```bash
hps_demo/
    hps_demo/
        __init__.py
    setup.py
```

We can now define DeepHyper search problems inside this directory, using `deephyper new-problem hps {name}`.

Let's set up an HPS problem called mnist_hps as follows:

```bash
cd hps_demo/hps_demo/
deephyper new-problem hps mnist_hps
```

A new HPS problem subdirectory should be in place. This is a Python subpackage containing sample code in the files `__init__.py`, `load_data.py`, `model_run.py`, and `problem.py`. Overall, your project directory should look like:

```bash
hps_demo/
    hps_demo/
        __init__.py
        mnist_hps/
            __init__.py
            load_data.py
            model_run.py
            problem.py
    setup.py
```



### TODO:

- Explain + test `hps/load_data.py`

- Explain + test `hps/mnist_mlp.py`

- Explain + test `hps/problem.py`

  
