#!/bin/bash
datasets=(MNIST FashionMNIST CIFAR10 CIFAR100)
epochs=(15 10 10 10)

conda activate vision
for ((i = 0; i < 4; i++)); do
	python3 train.py --model MLP --dataset "${datasets[$i]}" --epochs "${epochs[$i]}"
done