# VisionNETs
PyTorch implementations of popular vision neural networks

## Results

| Model   	| MNIST    | CIFAR-10 	| CIFAR-100	| FashionMNIST  | ImageNet  	|
|---------- |----------|----------	|----------	|--------------	|-------------- |
| MLP     	| 0.9800   | 0.5404   	| 0.2908   	| 0.8857 		|               |
| LeNet5  	| 0.9912   | 0.7366 	| 0.4089   	| 0.9109   		|               |
| AlexNet 	| 0.9914   | 0.7404   	| 0.3485	| 0.9058   		|               |
| ResNet  	|          |          	|          	|          		|               |
| VGG16  	| 0.9956   | 0.8696   	| 0.5581   	| 0.9390   		|               |
| GoogLeNet	|          |          	|          	|          		|               |

## Models

- [x] Multi-Layer Perceptron
- [x] [LeNet5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- [x] [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- [ ] [ResNet](https://arxiv.org/abs/1704.06904)
- [x] [VGG16](https://arxiv.org/abs/1505.06798)
- [ ] [GoogLeNet](https://arxiv.org/abs/1409.4842)

## Datasets

- [x] MNIST
- [x] CIFAR-10
- [x] CIFAR-100
- [x] FashionMNIST
- [x] ImageNet

## Replication Details

Any specific parameters used to achieve the results above:

| Model   	| MNIST    	| CIFAR-10 	| CIFAR-100	| FashionMNIST 	| ImageNet 	|
|---------	|--------  	|----------	|----------	|--------------	|----------	|
| MLP     	| 15 epochs	| 10 epochs | 10 epochs | 10 epochs	 	|          	|
| LeNet5  	| 10 epochs | 10 epochs | 15 epochs | 15 epochs     	|          	|
| AlexNet 	| 2 epochs	| 5 epochs 	| 5 epochs  | 5 epochs     	|          	|
| ResNet  	|        	|          	|           |              	|          	|
| VGG16 	| 5 epochs	| 20 epochs	| 10 epochs | 7 epochs 		|          	|
| GoogLeNet	|        	|          	|           |              	|          	|

## Pre-Trained Weights

Due to the size of the larger model weights, I had to move the pre-trained model checkpoints to a Google Drive folder.
You may access the pre-trained weights at the following link: [Google Drive](https://drive.google.com/drive/folders/12HhPDR_I2pdhZ5VBbv56mn2bZWVrdK1t?usp=share_link)
