#!/bin/bash
datasets=(MNIST FashionMNIST CIFAR10 CIFAR100)
epochs=(10 15 10 15)

conda activate vision
for ((i = 0; i < 4; i++)); do
	python3 train.py --model LeNet5 --dataset "${datasets[$i]}" --epochs "${epochs[$i]}"
done