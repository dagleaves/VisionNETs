# VisionNETs
PyTorch implementations of popular vision neural networks

## Results

| Model   	| MNIST    | FashionMNIST  | CIFAR-10 	| CIFAR-100	| ImageNet  	|
|--------- |----------|---------------|----------	|----------	|-------------- |
| MLP     	| 0.9800   | 0.8857 		| 0.5404   	    | 0.2908   	|               |
| LeNet5  	| 0.9912   | 0.9109   		| 0.7366 	      | 0.4089   	|               |
| AlexNet 	| 0.9947   | 0.9297   		| 0.8322   	    | 0.5505	|               |
| ResNet50 	| 0.9948   | 0.9167 		| 0.6046   	    | 0.4391   	|               |
| VGG16  	| 0.9956   | 0.9390   		| 0.8696   	    | 0.5581   	|               |
| GoogLeNet	| 0.9947   | 0.9236   		| 0.7995   	    | 0.4785   	|               |


## Replication Details

### Environment

First, you must obtain the necessary dependencies, which are provided as a conda environment
from the environment.yml file. With Anaconda or Miniconda installed, this can be imported
using the following command:

```shell
conda env create -f environment.yml
```

Then, activate the environment and the program will be ready to run.

```shell
conda activate visionnets
```

### Running the program

The full list of command-line arguments can be found with `python train.py --help`.
To replicate the exact results above, scripts can be found in the `scripts/` directory
for each model.

To train a new model, the base command is as follows:

```shell
python train.py --model MLP --dataset MNIST --epochs 10
```

This will automatically download the MNIST dataset if it is not already found in the 
`--data_dir` (default `data/`) and train the MLP model for 10 epochs and save the 
final model checkpoint to `--ckpt_dir` (default `checkpoints/`).

### Non-default parameters used

Any specific parameters that were used to achieve the results above:

| Model   	| MNIST    	| FashionMNIST 	| CIFAR-10 	| CIFAR-100	| ImageNet 	|
|---------	|--------  	|--------------	|----------	|----------	|----------	|
| MLP     	| 15 epochs	| 10 epochs	 	| 10 epochs | 10 epochs |          	|
| LeNet5  	| 10 epochs | 15 epochs     | 10 epochs | 15 epochs |          	|
| AlexNet 	| 10 epochs	| 15 epochs    	| 10 epochs | 15 epochs |          	|
| ResNet50 | 10 epochs	| 15 epochs    	| 10 epochs | 15 epochs |          	|
| VGG16 	| 5 epochs	| 7 epochs 		| 20 epochs	| 10 epochs |          	|
| GoogLeNet| 4 epochs 	| 10 epochs   	| 20 epochs	| 10 epochs |          	|

## Pre-Trained Weights

Due to the size of the larger model weights, I had to move the pre-trained model checkpoints to a Google Drive folder.
You may access the pre-trained weights at the following link: 
[Google Drive](https://drive.google.com/drive/folders/12HhPDR_I2pdhZ5VBbv56mn2bZWVrdK1t?usp=share_link)
