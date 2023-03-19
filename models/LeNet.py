import torch.nn as nn
import torch


class LeNet5(nn.Module):
    """
    Implementation of the LeNet5 architecture
        - Relu activation
        - MaxPool subsampling
        - Relu activation
    """
    def __init__(self, in_channels: int, out_features: int):
        super(LeNet5, self).__init__()

        padding = 2 if in_channels == 1 else 0
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6*in_channels, kernel_size=5, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6*in_channels, out_channels=16*in_channels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16*in_channels, out_channels=120*in_channels, kernel_size=5),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120*in_channels, out_features=84*in_channels),
            nn.ReLU(),
            nn.Linear(in_features=84*in_channels, out_features=out_features)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @classmethod
    def from_args(cls, args):
        dataset = args.dataset.lower()
        if dataset == 'mnist':
            return LeNet5(in_channels=1, out_features=10)
        elif dataset == 'cifar10':
            return LeNet5(in_channels=3, out_features=10)
        elif dataset == 'cifar100':
            return LeNet5(in_channels=3, out_features=100)
        else:
            raise NotImplementedError('LeNet5 not implemented for ' + args.dataset + ' dataset')
