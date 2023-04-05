#!/bin/bash
datasets=(MNIST FashionMNIST CIFAR10 CIFAR100)
epochs=(5 7 20 10)

conda activate vision
for ((i = 0; i < 4; i++)); do
	python3 train.py --model VGG16 --dataset "${datasets[$i]}" --epochs "${epochs[$i]}"
done