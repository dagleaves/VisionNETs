import torch.nn as nn
import torch


def conv1x1_block(in_channels: int, out_channels: int, stride: int = 1):
    """
    Basic 3x3 convolution block + batch norm (no relu)
    :param in_channels: channels going into the block
    :param out_channels: channels going out of the block
    :param stride: stride for the convolution layer
    :return: nn.Sequential
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
        nn.BatchNorm2d(out_channels)
    )


def conv3x3_block(in_channels: int, out_channels: int, stride: int = 1):
    """
    Basic 3x3 convolution block + batch norm (no relu)
    :param in_channels: channels going into the block
    :param out_channels: channels going out of the block
    :param stride: stride for the convolution layer
    :return: nn.Sequential

    NOTE: Padding is default set to 1 to preserve image width and height dims within blocks
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channels)
    )


class Bottleneck(nn.Module):
    """
    Basic ResNet block for the "deep" ResNet used in ResNet50
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = conv1x1_block(in_channels, in_channels, stride=stride)
        self.conv2 = conv3x3_block(in_channels, in_channels)
        self.conv3 = conv1x1_block(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # 1x1 convolution - downsample if block 3_1, 4_1, or 5_1 - dimension bottleneck squeeze
        out = self.conv1(x)
        out = self.relu(out)

        # 3x3 convolution
        out = self.conv2(out)
        out = self.relu(out)

        # 1x1 dimension increase
        out = self.conv3(out)
        # NOTE: No relu here, must combine with residual connection first

        # If block downsampled input, downsample the identity to match
        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual/skip connection
        out += identity
        out = self.relu(out)
        return out


def bottleneck_layer(channels: int, width: int, stride: int):
    input_channels = channels * 2 if stride != 1 else channels # num input channels from previous layer
    expansion = 4
    downsample = conv1x1_block(input_channels, channels * expansion, stride)
    blocks = [
        Bottleneck(input_channels, channels * expansion, stride, downsample)
    ]
    channels *= expansion
    for _ in range(1, width):
        blocks.append(Bottleneck(channels, channels))
    return nn.Sequential(*blocks)


class ResNet50(nn.Module):
    """
    Implementation of the ResNet50 architecture
    From https://arxiv.org/pdf/1512.03385.pdf
    """
    def __init__(self, in_channels: int, out_features: int, in_features: int = 224):
        super(ResNet50, self).__init__()
        padding = 3 if in_channels == 3 else 5
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            bottleneck_layer(64, 3, 1),
            bottleneck_layer(128, 4, 2),
            bottleneck_layer(256, 6, 2),
            bottleneck_layer(512, 3, 2)
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
            raise NotImplementedError('ResNet50 not implemented for ' + args.dataset + ' dataset')