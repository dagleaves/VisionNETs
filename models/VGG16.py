import torch.nn as nn
import torch


def conv3x3(in_channels: int, out_channels: int):
    """ Generic 3x3 conv2d factory """
    padding = 3 if in_channels == 1 else 1  # convert 28x28 MNIST to 32x32
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding)


def conv_block(in_channels: int, out_channels: int, width: int):
    """
    Base VGG conv block
    :param in_channels: channels going into the block
    :param out_channels: channels going out of the block (and used within)
    :param width: number of convolution layers within the block
    :return: nn.Sequential
    """
    block = []
    for _ in range(width):
        block.extend([
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        in_channels = out_channels  # make sure only first layer uses initial in_channels

    block.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*block)


class VGG16(nn.Module):
    """
    Implementation of the VGG16 architecture
    From https://arxiv.org/pdf/1409.1556.pdf
    """
    def __init__(self, in_channels: int, out_features: int, dropout: float = 0.5, in_features: int = 224):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, 64, 2),
            conv_block(64, 128, 2),
            conv_block(128, 256, 3),
            conv_block(256, 512, 3),
            conv_block(512, 512, 3)
        )

        classifier_dim = 7 if in_features == 224 else 1
        self.avgpool = nn.AdaptiveAvgPool2d((classifier_dim, classifier_dim))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * classifier_dim * classifier_dim, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=out_features)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @classmethod
    def from_args(cls, args):
        dataset = args.dataset.lower()
        if dataset == 'mnist':
            return VGG16(in_channels=1, out_features=10, in_features=32)
        elif dataset == 'cifar10':
            return VGG16(in_channels=3, out_features=10, in_features=32)
        elif dataset == 'cifar100':
            return VGG16(in_channels=3, out_features=100, in_features=32)
        elif dataset == 'fashionmnist':
            return VGG16(in_channels=1, out_features=10, in_features=32)
        elif dataset == 'imagenet':
            return VGG16(in_channels=3, out_features=1000, in_features=224)
        else:
            raise NotImplementedError('VGG16 not implemented for ' + args.dataset + ' dataset')