import torch.nn as nn
import torch


def conv3x3(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


def conv_block(in_channels: int, out_channels: int, width: int):
    block = [conv3x3(in_channels, out_channels), nn.ReLU(inplace=True)]
    for _ in range(width - 1):
        block.extend([conv3x3(out_channels, out_channels), nn.ReLU(inplace=True)])
    block.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*block)


class VGG16(nn.Module):
    """
    Implementation of the VGG16 architecture
    From https://arxiv.org/pdf/1409.1556.pdf
    """
    def __init__(self, in_channels: int, out_features: int):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, 64, 2),
            conv_block(64, 128, 2),
            conv_block(128, 256, 3),
            conv_block(256, 512, 3),
            conv_block(512, 512, 3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=out_features)
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
            return VGG16(in_channels=1, out_features=10)
        elif dataset == 'cifar10':
            return VGG16(in_channels=3, out_features=10)
        elif dataset == 'cifar100':
            return VGG16(in_channels=3, out_features=100)
        elif dataset == 'fashionmnist':
            return VGG16(in_channels=1, out_features=10)
        elif dataset == 'imagenet':
            return VGG16(in_channels=3, out_features=1000)
        else:
            raise NotImplementedError('VGG16 not implemented for ' + args.dataset + ' dataset')