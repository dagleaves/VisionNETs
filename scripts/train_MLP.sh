#!/bin/sh
python train.py --model MLP --dataset MNIST --n_epochs 2
python train.py --model MLP --dataset CIFAR10 --n_epochs 5

