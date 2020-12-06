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
