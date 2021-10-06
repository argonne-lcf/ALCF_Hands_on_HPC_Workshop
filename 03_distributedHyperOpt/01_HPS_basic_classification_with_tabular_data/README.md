# Hyperparameter Search for Classification with Tabular Data

In this tutorial we will present how to run an hyperparameter search with DeepHyper on the **ThetaGPU** system. The machine learning task is about detecting if a person has an heart desease. To do so, we will use tabular data and a binary-classification setting.

To execute this tutorial it is possible to run an interactive Jupyter notebook locally or in an interactive session on ThetaGPU.

## Local execution

Install DeepHyper with the `analytics` option then launch Jupyter:

```console
jupyter notebook
```

## ThetaGPU execution

1. From a `thetalogin` node: `ssh thetagpusn1` to login to a ThetaGPU service node.
2. From `thetagpusn1`, start an interactive job:

```bash
(thetagpusn1) $ qsub -I -A datascience -n 1 -q training-gpu -t 60
Job routed to queue "full-node".
Wait for job 10003623 to start...
Opening interactive session to thetagpu21
```

3. From the ThetaGPU compute node, execute the `launch-jupyter-notebook.sh` script:

```bash
(thetagpu21) $ ./launch-jupyter-notebook.sh
```

Take note of the output URL and output ssh command of the form:

```bash
# URL
http://localhost:8888/?token=df11ba29aac664173832b98d1d4b3b96ee0f050992ae6591
# SSH Command
ssh -tt -L 8888:localhost:8888 -L 8265:localhost:8265 regele@theta.alcf.anl.gov "ssh -L 8888:localhost:8888 -L 8265:localhost:8265 thetagpu05"
```

4. Leave the interactive session open and in a new terminal window of your laptop execute the SSH command to link local port to the ThetaGPU compute node.

5. Open the Jupyter URL in a local browser.
