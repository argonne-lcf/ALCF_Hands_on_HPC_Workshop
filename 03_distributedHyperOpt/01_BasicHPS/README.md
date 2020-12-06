# Hyperparameter Search for Deep Learning

**TODO**: 

- Include instructions for port-forwarding to connect to jupyter from local machine
- Include code from `load_data.py` and explain
- Include code from `model_run.py` and explain
- Include code from `problem.py` and explain
- Include links to `DeepHyper`, `Balsam` (github + documentation)
- Include more detail throughout, walk through code blocks
  - Explain the hyperparameters in `problem.py`
  - Include section that tests each of the `load_data.py`, `model_run.py`, and `problem.py` scripts individually to make sure they run
- The `deephyper/0.2.1` module seems broken somehow, workaround for now:
  - Issues encountered when trying to install deephyper with horovod, but should run without issue using analytics and balsam.

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

## Environment setup on ALCF's`Theta`:

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

### Load and split MNIST data in to distinct train / test sets

We can load and split the data as follows, using the code from [03_distributedHyperOpt/01_BasicHPS/load_data.py](load_data.py):

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

### Defining the Search Space

We include below the code for building, training, and evaluating the trained MNIST model. Note that this is indentical to the [`03_distributedHyperOpt/01_BasicHPS/model_run.py](model_run.py).

For simplicity we look at a multi-layer perceptron with two hidden layers, built using [tf.keras.Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential):

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

HISTORY = None

# Define default values of hyperparameters to use if None are specified
POINT = { 
    'epochs': 2,
    'units1': 8,
    'units2': 16,
    'dropout1': 0.,
    'dropout2': 0.,
    'batch_size': 16,
    'activation': 'relu',
    'optimizer': 'SGD',
}

def run(point: dict = None):
    global HISTORY
    if point is None:
        point = POINT
        
    print(point)
    
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]
    
    epochs = point.get('epochs', None)
    units1 = point.get('units1', None)
    units2 = point.get('units2', None)
    dropout1 = point.get('dropout1', None)
    dropout2 = point.get('dropout2', None)
      
    model = tf.keras.Sequential([
        layers.Dense(point['units1'], activation=point['activation']),
        layers.Dropout(point['dropout1']),
        layers.Dense(point['units2'], activation=point['activation']),
        layers.Dropout(point['dropout2']),
        layers.Dense(10),
    ])
    
    model.compile(
        optimizer=point['optimizer'],
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
      
    )
    
    # Train the model
    _ = model.fit(
        x_train,
        y_train,
        batch_size=point['batch_size'],
        epochs=point['epochs'],
        verbose=1,
        callbacks=[keras.callbacks.EarlyStopping(monitor='loss', verbose=1)],
        # We need to pass some validation data
        # for monitoring the validation loss and metrics
        # at the end of each epoch
        validation_data=(x_val, y_val),
    )
    
    # Evaluate the trained model on the test set
    score = model.evaluate(x_test, y_test, verbose=2)
    print(f'test_loss, test_accuracy: {score[0]}, {score[1]}')
    
    return score[1]

if __name__ == '__main__':
    from problem import Problem
    point = Problem.starting_point_asdict[0]
    run(point)
```

## Launch an Experiment

The deephyper Theta module has a convenience script included for quick generation of DeepHyper Async Bayesian Model Search (AMBS) search jobs. Simply pass the paths to the `model_run.py` script (containing the `run()` function), and the `problem.py` file (containing the `HpProblem`) as follows:

```bash
deephyper balsam-submit hps mnist-demo -p problem.py -r model_run.py \
    -t 20 -q debug-cache-quad -n 2 -A datascience -j serial
```



### Monitor Execution and Check Results

You can use Balsam to watch when the experiement starts running and track how many models are running in realtime. Once the ambs task is RUNNING, the `bcd` command line tool provides a convenient way to jump to the working directory, which will contain the DeepHyper log and search results in CSV or JSON format.

Notice the objective value in the second-to-last column of the `results.csv` file:

```bash
 balsam ls --wf mnist-demo
                              job_id |       name |   workflow | application |   state
--------------------------------------------------------------------------------------
b1dd0a04-dbd5-4601-9295-7465abe6b794 | mnist-demo | mnist-demo | AMBS        | CREATED

. bcd b1dd  # Note: 'b1dd' is the prefix of the `job_id` above ^
```

### DeepHyper analytics

Run:

```bash
deephyper-analytics hps -p results.csv
```

Start `jupyter`:

```bash
jupyter notebook
```
