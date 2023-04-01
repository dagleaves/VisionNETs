import torch.nn as nn
import torch


def conv1x1_block(in_channels: int, out_channels: int, stride: int = 1, padding: int = 1):
    """
    Basic 3x3 convolution block + batch norm (no relu)
    :param in_channels: channels going into the block
    :param out_channels: channels going out of the block
    :param stride: stride for the convolution layer
    :param padding: padding for the convolution layer
    :return: nn.Sequential
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels)
    )


def conv3x3_block(in_channels: int, out_channels: int, stride: int = 1):
    """
    Basic 3x3 convolution block + batch norm (no relu)
    :param in_channels: channels going into the block
    :param out_channels: channels going out of the block
    :param stride: stride for the convolution layer
    :return: nn.Sequential
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride),
        nn.BatchNorm2d(out_channels)
    )


class Bottleneck(nn.Module):
    """
    Basic ResNet block for the "deep" ResNet used in ResNet50
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv1 = conv1x1_block(in_channels, in_channels, stride)
        self.conv2 = conv3x3_block(in_channels, in_channels)
        self.conv3 = conv1x1_block(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # 1x1 convolution - downsample if block 3_1, 4_1, or 5_1 - dimension bottleneck squeeze
        x = self.conv1(x)
        x = self.relu(x)

        # 3x3 convolution
        x = self.conv2(x)
        x = self.relu(x)

        # 1x1 dimension increase
        x = self.conv3(x)
        x = self.relu(x)

        # If block downsampled input, downsample the identity to match
        if self.stride != 1:
            identity = conv1x1_block(self.in_channels, self.out_channels, self.stride)

        # Residual/skip connection
        x += identity
        x = self.relu(x)
        return x


def bottleneck_layer(in_channels: int, block_channels: int, out_channels: int, width: int):
    stride = 2 if in_channels != block_channels else 1
    blocks = [
        Bottleneck(block_channels, out_channels, stride)
    ]
    for _ in range(1, width):
        blocks.append(Bottleneck(block_channels, out_channels))
    return nn.Sequential(*blocks)


class ResNet50(nn.Module):
    """
    Implementation of the ResNet50 architecture
    From https://arxiv.org/pdf/1512.03385.pdf
    """
    def __init__(self, in_channels: int, out_features: int, in_features: int = 224):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            bottleneck_layer(64, 64, 256, 3),
            bottleneck_layer(256, 128, 512, 4),
            bottleneck_layer(512, 256, 1024, 6),
            bottleneck_layer(1024, 512, 2048, 3)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_features=2048, out_features=out_features)

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
            return ResNet50(in_channels=1, out_features=10, in_features=32)
        elif dataset == 'cifar10':
            return ResNet50(in_channels=3, out_features=10, in_features=32)
        elif dataset == 'cifar100':
            return ResNet50(in_channels=3, out_features=100, in_features=32)
        elif dataset == 'fashionmnist':
            return ResNet50(in_channels=1, out_features=10, in_features=32)
        elif dataset == 'imagenet':
            return ResNet50(in_channels=3, out_features=1000, in_features=224)
        else:
            raise NotImplementedError('VGG16 not implemented for ' + args.dataset + ' dataset')