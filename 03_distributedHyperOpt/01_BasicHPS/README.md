# Hyperparameter Search for Deep Learning

**TODO**: 

- [ ] Include instructions for port-forwarding to connect to jupyter from local machine

- [x] Include code from `load_data.py` and explain

- [x] Include code from `model_run.py` and explain

- [x] Include code from `problem.py` and explain

- [ ] Include links to `DeepHyper`, `Balsam` (github + documentation)

- [ ] Include more detail throughout, walk through code blocks

- [ ] Explain the hyperparameters in `problem.py`

- [x] Include section that tests each of the `load_data.py`, `model_run.py`, and `problem.py` scripts individually to make sure they run
- [ ] Remove `model_run.py` test (tensorflow isn't supposed to run on login nodes??)

- [ ] The `deephyper/0.2.1` module seems broken somehow, workaround for now:
  - [ ] Issues encountered when trying to install deephyper with horovod, but should run without issue using analytics and balsam.

```bash
module load postgresql
module load miniconda-3
conda create -p dh-env
conda activate dh-env
conda install gxx_linux-64 gcc_linux-64
conda install tensorflow -c intel
# DeepHyper + Analytics Tools (Parsing logs, Plots, Notebooks)
pip install deephyper[analytics,balsam]
# or DeepHyper + Analytics Tools (Parsing logs, Plots, Notebooks) + Horovod
pip install deephyper[analytics,hvd,balsam]
```

---

Every DeepHyper search requires at least 2 Python objects as input:

- `run`: Your "black-box" function returning the objective value to be maximized
- `Problem`: an instance of `deephyper.problem.BaseProblem` which defines the search space of input parameters to `run`.

We will illustrate DeepHyper HPS for the MNIST dataset, with a goal of tuning the hyperparameters to maximize the classification accuracy.

## Environment setup on `Theta` @ALCF:

To start on Theta, let's set up  a clean workspace for the HPS:

```bash
# Create a new workspace with Balsam DB
module unload balsam  # Already included in  DeepHyper-0.2.1
module load deephyper/0.2.1  # Includes Balsam, TensorFlow, Keras, etc...
rm -r ~/.balsam  # reset default settings (for now)
```

If you haven't already:

```bash
git clone https://github.com/argonne-lcf/sdl_ai_workshop
```

Navigate into the BasicHPS directory:

```bash
cd sdl_ai_workshop/03_distributedHyperOpt/01_BasicHPS
git pull  # make sure you're up to date
```



We can now our search scaled to run parallel model evaluations across multiple nodes of Theta.

First, create a Balsam database:

```bas
balsam init db
```

Start and connecto to the `db` database:

```bash
source balsamactivate db
```

## Setup the HPS

### Load data

We can load and split the MNIST data into distinct train / test sets using the code from [`load_data.py`](load_data.py):

```python
import tensorflow as tf

def load_data():
    """Returns MNIST data as `(x_train, y_train), (x_test, y_test)`."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Flatten the data and normalize
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    
    print(f'(x_train, y_train): ({x_train.shape}, {y_train.shape})')'
    print(f'(x_test, y_test): ({x_test.shape}, {y_test.shape})')'
    
    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    load_data()
```

Running this interactively should produce:

```bash
python3 load_data.py
```

```bash
(x_train, y_train): ((60000, 784), (60000,))
(x_test, y_test): ((10000, 784), (10000,))
```

### Model Implementation (with [tf.Keras](https://www.tensorflow.org/api_docs/python/tf/keras))

We include below the code for building, training, and evaluating the trained MNIST model. Note that this is indentical to the [`model_run.py`](model_run.py). For simplicity we look at a multi-layer perceptron with two hidden layers, built using [tf.keras.Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential).

```python
import os
import sys
import tensorflow as tf

# Identify current working directory and inject it into `sys.path`
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from load_data import load_data

keras = tf.keras
layers = tf.keras.layers

def run(point: dict = None):
    print(point)
    
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    epochs = 10  # Fixed num of epochs (for now)
    optimizer = point.get('optimizer', None)
    batch_size = point.get('batch_size', None)
    activation = point.get('activation', None)

    units1 = point.get('units1', None)
    units2 = point.get('units2', None)
    dropout1 = point.get('dropout1', None)
    dropout2 = point.get('dropout2', None)

    model = tf.keras.Sequential([
        layers.Dense(units1, activation=activation),
        layers.Dropout(dropout1),
        layers.Dense(units2, activation=activation),
        layers.Dropout(dropout2),
        layers.Dense(10)
    ])

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    _ = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        # We need to pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_val, y_val),
    )
    score = model.evaluate(x_test, y_test, verbose=2)
    print(f'test_loss, test_accuracy: {score[0]}, {score[1]}')

    return score[1]


if __name__ == '__main__':
    from problem import Problem
    point = Problem.starting_point_asdict[0]
    run(point)
```

We can test that our environment is setup correctly by trying to run this script:

```bash
python3 model_run.py
```

Should output:

```bash
{'activation': 'relu', 'batch_size': 8, 'dropout1': 0.0, 'dropout2': 0.0, 'epochs': 5, 'optimizer': 'SGD', 'units1': 1, 'units2': 2}
(x_train, y_train): ((60000, 784), (60000,))
(x_test, y_test): ((10000, 784), (10000,))
2020-12-06 14:45:45.323672: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-12-06 14:45:45.337630: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7fb2d7f26e80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-12-06 14:45:45.337647: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Epoch 1/5
6250/6250 [==============================] - 3s 545us/step - loss: 2.3015 - accuracy: 0.1131 - val_loss: 2.3021 - val_accuracy: 0.1064
Epoch 2/5
6250/6250 [==============================] - 3s 547us/step - loss: 2.3013 - accuracy: 0.1136 - val_loss: 2.3025 - val_accuracy: 0.1064
Epoch 3/5
6250/6250 [==============================] - 3s 549us/step - loss: 2.3014 - accuracy: 0.1136 - val_loss: 2.3023 - val_accuracy: 0.1064
Epoch 4/5
6250/6250 [==============================] - 3s 550us/step - loss: 2.3014 - accuracy: 0.1136 - val_loss: 2.3027 - val_accuracy: 0.1064
Epoch 5/5
6250/6250 [==============================] - 3s 545us/step - loss: 2.3014 - accuracy: 0.1136 - val_loss: 2.3020 - val_accuracy: 0.1064
313/313 - 0s - loss: 2.3011 - accuracy: 0.1135
test_loss, test_accuracy: 2.301095485687256, 0.11349999904632568
```



### Defining the Search Space



The `run` function shown below expects a hyperparameter dictionary with keys shown in `POINT` below.

We define acceptable ranges for these hyperparameters with the `Problem` object inside [`problem.py`](problem.py) . Hyperparameter ranges are defined using the following syntax:

- Discrete integer ranges are generated from a tuple `(lower: int, upper: int)`
- Continuous prarameters are generated from  a tuple `(lower: float, upper: float)`
- Categorical or nonordinal hyperparameter ranges can be given as a list of possible values `[val1, val2, ...]`

#### Problem definition

We include below the complete problem definition from [`problem.py`](problem.py) which is responsible for defining the search space in terms of the hyperparameter regions.

```python
from deephyper.problem import HpProblem

Problem = HpProblem()
Problem.add_dim('units1', (1, 64))
Problem.add_dim('units2', (1, 64))
Problem.add_dim('dropout1', (0.0, 1.0))
Problem.add_dim('dropout2', (0.0, 1.0))
Problem.add_dim('batch_size', (5, 500))
Problem.add_dim('learning_rate', (0.0, 1.0))
Problem.add_dim('activation', ['relu', 'elu', 'selu', 'tanh'])
Problem.add_dim('optimizer', ['Adam', 'RMSprop', 'SGD', 'Nadam', 'Adagrad'])

Problem.add_starting_point(
    units1=16,
    units2=32,
    dropout1=0.0,
    dropout2=0.0,
    batch_size=16,
    activation='relu',
    optimizer='SGD',
    learning_rate=0.001,
)


if __name__ == "__main__":
    print(Problem)
```

## Launch an Experiment

The deephyper Theta module has a convenience script included for quick generation of DeepHyper Async Bayesian Model Search (AMBS) search jobs. Simply pass the paths to the `model_run.py` script (containing the `run()` function), and the `problem.py` file (containing the `HpProblem`) as follows:

```bash
deephyper balsam-submit hps mnist-demo -p problem.py -r model_run.py \
    -t 20 -q debug-cache-quad -n 2 -A datascience -j mpi
```

### Monitor Execution and Check Results

You can use Balsam to watch when the experiement starts running and track how many models are running in realtime. Once the ambs task is RUNNING, the `bcd` command line tool provides a convenient way to jump to the working directory, which will contain the DeepHyper log and search results in CSV or JSON format.

Notice the objective value in the second-to-last column of the `results.csv` file:

```bash
 balsam ls --wf mnist-demo
 ```
 ```bash
                              job_id |       name |   workflow | application |   state
--------------------------------------------------------------------------------------
b1dd0a04-dbd5-4601-9295-7465abe6b794 | mnist-demo | mnist-demo | AMBS        | CREATED
```
```bash
# We can jump directly to the working directory containing the DeepHyper log
. bcd b1dd  # Note: 'b1dd' is the prefix of the `job_id` above; yours will be different
```

### DeepHyper analytics

Run:

```bash
deephyper-analytics hps -p results.csv
```

We can open a jupyter notebook on a `thetalogin` node 

```bash
jupyter notebook
```
