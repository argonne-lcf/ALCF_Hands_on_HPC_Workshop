# Hyperparameter search for Classification with Tabular Data

In this tutorial we will present how to run an hyperparameter search with DeepHyper on the **ThetaGPU** system. The machine learning task is about detecting if a person has an heart desease. To do so, we will use tabular data and a binary-classification setting.

To execute this tutorial it is possible to run an interactive Jupyter notebook locally or in an interactive session on ThetaGPU. For the interactive session on ThetaGPU please follow the [first section](##ssh-tunneling-for-jupyter-notebooks).

## SSH Tunneling for Jupyter Notebooks

1. From a `thetalogin` node: `ssh thetagpusn1` to login to a ThetaGPU service node.
2. From `thetagpusn1`, start an interactive job (make sure to note which ThetaGPU node the job gets routed to, `thetagpu21` in this example):

```bash
(thetagpusn1) $ qsub -I -A datascience -n 1 -q full-node -t 01:00 --interactive
Job routed to queue "full-node".
Wait for job 10003623 to start...
Opening interactive session to thetagpu21
```

3. From the ThetaGPU compute node, start a `jupyter` notebook.:

```bash
(thetagpu21) $ jupyter notebook&
```

Take note of the port number `N` that this Jupyter process starts on. By default, the port
is 8888. But if that port is in use, or the `--port X` flag is specified, it will be
different.

4. From a new terminal (on your local machine):

```bash
$ export PORT_NUM=8889  # port on your local machine; picking a number other than the default 8888 is recommended
$ ssh -L $PORT_NUM:localhost:8888 username@theta.alcf.anl.gov
(thetalogin) $ ssh -L 8888:localhost:N thetagpu21    # where N is the port number on the ThetaGPU compute node, noted in the previous step
```

5. Navigating to `localhost:8889` (or whatever port number you chose above) on your local machine's browser should then establish a connection to the Jupyter backend!

6. Finally, open the `02_HPS_basic_classification_with_tabular_data/02_HPS_basic_classification_with_tabular_data.ipynb` notebook to follow the interactive tutorial.

## Submission Script

Start by loading the DeepHyper module:

```bash
module load ...
```

Then submit the prepared DeepHyper job scrip:

```bash
cd 02_HPS_basic_classification_with_tabular_data/
qsub deephyper-job.qsub
```

Once the job is over look at the results:

```bash
deephyper-analytics topk -k 3 02_HPS_basic_classification_with_tabular_data/results.csv
```
