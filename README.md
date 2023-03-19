# VisionNETs
PyTorch implementations of popular vision neural networks

## Results

| Model   	| MNIST 	| CIFAR-10 	| CIFAR-100	| ImageNet  | FashionMNIST  |
|---------- |----------	|----------	|----------	|----------	|-------------- |
| MLP     	| 0.9671    | 0.4717   	| 0.2098   	|          	|               |
| LeNet5  	| 0.9865 	| 0.6800 	| 0.3172   	|          	|               |
| AlexNet 	| 0.9831    |          	|           |          	|               |
| ResNet  	|           |          	|          	|          	|               |
| VGG16  	|           |          	|          	|          	|               |
| GoogLeNet	|           |          	|          	|          	|               |

## Models

- [x] Multi-Layer Perceptron
- [x] [LeNet5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- [ ] [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- [ ] [ResNet](https://arxiv.org/abs/1704.06904)
- [ ] [VGG16](https://arxiv.org/abs/1505.06798)
- [ ] [GoogLeNet](https://arxiv.org/abs/1409.4842)

## Datasets

- [x] MNIST
- [x] CIFAR-10
- [x] CIFAR-100
- [ ] FashionMNIST
- [ ] ImageNet

## Replication Details

Any specific parameters used to achieve the results above:

| Model   	| MNIST    	| CIFAR-10 	| CIFAR-100	| FashionMNIST 	| ImageNet 	|
|---------	|--------  	|----------	|----------	|--------------	|----------	|
| MLP     	| 2 epochs	| 5 epochs 	| 5 epochs 	|              	|          	|
| LeNet5  	| 2 epochs 	| 5 epochs 	| 5 epochs 	|              	|          	|
| AlexNet 	|        	|          	|           |              	|          	|
| ResNet  	|        	|          	|           |              	|          	|
| VGG16 	|        	|          	|           |              	|          	|
| GoogLeNet	|        	|          	|           |              	|          	|
