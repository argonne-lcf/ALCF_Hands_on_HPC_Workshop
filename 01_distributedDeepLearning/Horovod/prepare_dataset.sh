#!/bin/sh
# Please run this script on a login node
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
[ -e ~/.keras/ ] || mkdir ~/.keras/
[ -e ~/.keras/datasets/ ] || mkdir ~/.keras/datasets
mv mnist.npz ~/.keras/datasets
