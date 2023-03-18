#!/bin/sh
python train.py --model LeNet5 --dataset MNIST --n_epochs 2
python train.py --model LeNet5 --dataset CIFAR10 --n_epochs 5
