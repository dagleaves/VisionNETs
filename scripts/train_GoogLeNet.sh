#!/bin/bash
datasets=(MNIST FashionMNIST CIFAR10 CIFAR100)
epochs=(4 10 20 10)

conda activate vision
for ((i = 0; i < 4; i++)); do
	python3 train.py --model GoogLeNet --dataset "${datasets[$i]}" --epochs "${epochs[$i]}"
done