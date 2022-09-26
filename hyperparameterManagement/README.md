[<img align=right src="01_HPS_basic_classification_with_tabular_data/Figures/Deephyper_transparent.png" width=25%>](https://github.com/deephyper/deephyper)

# Scalable Neural Architecture and Hyperparameter Search for Deep Neural Networks

[DeepHyper](https://github.com/deephyper/deephyper) is a distributed machine learning (AutoML) package for automating the developement of deep neural networks for scientific applications. It can run on a single laptop as well as on 1,000 of nodes.

It comprises different tools such as:

* Optimizing hyper-parameters for a given black-box function.
* Neural architecture search to discover high-performing deep neural network with variable operations and connections.
* Automated machine learning, to easily experiment many learning algorithms from Scikit-Learn.

DeepHyper provides an infrastructure that targets experimental research in NAS and HPS methods, scalability, and portability across diverse infrastructures (cloud, laptop, supercomputers). It comprises three main modules:

* `deephyper.problem`: Tools for defining neural architecture and hyper-parameter search problems.
* `deephyper.evaluator`: A simple interface to dispatch model evaluation tasks. Implementations range from subprocess for laptop experiments to ray for large-scale runs on HPC systems.
* `deephyper.search`: Search methods for NAS and HPS. By extending the generic Search class, one can easily add new NAS or HPS methods to DeepHyper.

Today, we will be introducing you to some of these features.

## Tutorials

Two tutorials are presented during this workshop:

1. Hyperparameter Search for Classification with Tabular Data
2. From Neural Architecture Search to Automated Deep Ensemble with Uncertainty Quantification

The first one is very light in computations, it can be executed in a few minutes from a local laptop and can also be tested on the ThetaGPU system. The second tutorial is more heavy in computation and provides some of our latest research.

## What else?

While the hands-on will be covering a few features on small scale problems, users should also be aware of the different applications on which DeepHyper was used. For example, neural architecture search has been used to _discover_ novel neural networks for problems as diverse as predicting cancer drug synergy [2] using data from the National Cancer Institute (NCI) and for forecasting the global sea-surface temperature using satellite data from NOAA [3].

The [DeepHyper documentation](https://deephyper.readthedocs.io/) can also be accessed to learn more about the software.

Do not hesitate to contact us and join our Slack workspace if you need more guidance.

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
