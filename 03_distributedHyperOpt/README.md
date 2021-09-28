<center><img src="Figures/Deephyper.png" height="200"></center>

# Scalable Neural Architecture and Hyperparameter Search for Deep Neural Networks

DeepHyper is a scalable automated machine learning (AutoML) package for developing deep neural networks for scientific applications. It comprises two components:

1. Hyperparameter Search (HPS): optimizing hyperparameters for a given reference model

2. Neural Architecture Search (NAS): fully-automated search for high-performing deep neural network architectures

In today's discussions, we will be introducing you to HPS for arbitrary data-driven modeling tasks ranging from simple linear and tree-based models to full-blown deep learning frameworks.

## Tutorials


## What else?

While the hands-on will be covering solely HPS today, users should also be aware of **NAS**: DeepHypers neural architecture search framework. NAS has been used to _discover_ novel neural networks for problems as diverse as predicting cancer drug synergy [2] using data from the National Cancer Institute (NCI) and for forecasting the global sea-surface temperature using satellite data from NOAA [3]. NAS interprets a neural network as a directed acyclic graph with its various nodes representing potential structural choices such as whether a skip connection should be used or if a a certain nonlinear activation should be applied. Users are encouraged to visit `https://deephyper.readthedocs.io/en/latest/tutorials/nas.html` for examples and tutorials.


## References

### 2021

* Egele, Romain, et al. “AgEBO-Tabular: Joint Neural Architecture andHyperparameter Search with Autotuned Data-Parallel Trainingfor Tabular Data.” In SC21: International Conference for HighPerformance Computing, Networking, Storage and Analysis. 2021.

### 2020

* Maulik, Romit, et al. “Recurrent neural network architecture search for geophysical emulation.” SC20: International Conference for High Performance Computing, Networking, Storage and Analysis. IEEE, 2020.

* Jiang, Shengli, and Prasanna Balaprakash. “Graph Neural Network Architecture Search for Molecular Property Prediction.” 2020 IEEE International Conference on Big Data (Big Data). IEEE, 2020.

### 2019

* Balaprakash, Prasanna, et al. “Scalable reinforcement-learning-based neural architecture search for cancer deep learning research.” Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis. 2019.

### 2018

* Balaprakash, Prasanna, et al. “DeepHyper: Asynchronous hyperparameter search for deep neural networks.” 2018 IEEE 25th international conference on high performance computing (HiPC). IEEE, 2018.