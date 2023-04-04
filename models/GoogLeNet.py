import torch.nn as nn
import torch
from collections import namedtuple


# Used for checking if model output has to use special loss calculation for the aux logits
GoogLeNetOutput = namedtuple('GoogLeNetOutput', ['logits', 'aux_logits1', 'aux_logits2'])


# Special GoogLeNet auxiliary loss calculation procedure
def gnet_loss(output, target, criterion):
    assert isinstance(output, GoogLeNetOutput), 'Input must be a GoogLeNet named tuple'
    main_loss = criterion(output.logits, target)
    if output.aux_logits1 is None or output.aux_logits2 is None:
        return main_loss
    aux1_loss = criterion(output.aux_logits1, target)
    aux2_loss = criterion(output.aux_logits2, target)
    return main_loss + (0.3 * aux1_loss) + (0.3 * aux2_loss)    # 0.3 discount weight from paper


def convblock(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
    """ Generic conv2d block factory """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class Inception(nn.Module):
    """
    Fundamental GoogLeNet Inception layer
    Padding is used to ensure image dimensions do not change within inception block
    """
    def __init__(self,
                 in_channels: int,
                 ch_1x1: int,
                 ch_3x3red: int,
                 ch_3x3: int,
                 ch_5x5red: int,
                 ch_5x5: int,
                 pool_proj: int
                 ):
        super(Inception, self).__init__()
        self.lane1 = convblock(in_channels, ch_1x1, kernel_size=1)
        self.lane2 = nn.Sequential(
            convblock(in_channels, ch_3x3red, kernel_size=1),
            convblock(ch_3x3red, ch_3x3, kernel_size=3, padding=1)
        )
        # NOTE: Although the paper says to use a kernel size 5 in Lane 3,
        # many implementations use kernel size 3, as the released implementation uses this.
        # I am not sure if there is a difference in performance, but it is a known issue.
        self.lane3 = nn.Sequential(
            convblock(in_channels, ch_5x5red, kernel_size=1),
            convblock(ch_5x5red, ch_5x5, kernel_size=5, padding=2)
        )
        self.lane4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            convblock(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        lane1 = self.lane1(x)
        lane2 = self.lane2(x)
        lane3 = self.lane3(x)
        lane4 = self.lane4(x)
        return torch.cat([lane1, lane2, lane3, lane4], 1)


class AuxClassifier(nn.Module):
    """
    Auxiliary classifier used for regularization during training
    """
    def __init__(self, in_channels, out_features):
        super(AuxClassifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = convblock(in_channels, 128, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(128*4*4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(1024, out_features)
        )

    def forward(self, x):
        out = self.avgpool(x)
        out = self.conv(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class GoogLeNet(nn.Module):
    """
    Implementation of the GoogLeNet architecture
    From https://arxiv.org/pdf/1409.4842.pdf
    """
    def __init__(self, in_channels: int, out_features: int,):
        super(GoogLeNet, self).__init__()
        in_padding = 3 if in_channels == 3 else 5   # Pad 28x28 to 32x32
        self.conv1 = convblock(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=in_padding)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # The depth is a bit confusing at first, but looking at the 3x3reduction column explains the extra 1x1 conv here
        self.conv2 = convblock(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        self.conv3 = convblock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(1024, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.classifier = nn.Linear(in_features=1024, out_features=out_features)

        # Auxiliary classifiers
        self.aux1 = AuxClassifier(512, out_features)    # Output from inception4a
        self.aux2 = AuxClassifier(528, out_features)    # Output from inception4d

    def forward(self, x):
        out = self.conv1(x)     # 112x112 or 16x16 -> 64 channels
        out = self.maxpool1(out)    # 56x56 or 8x8
        out = self.conv2(out)   # 64 channels
        out = self.conv3(out)   # 192 channels
        out = self.maxpool2(out)    # 28x28 or 4x4
        out = self.inception3a(out)     # 256 channels
        out = self.inception3b(out)     # 480 channels
        out = self.maxpool3(out)    # 14x14 or 2x2
        out = self.inception4a(out)     # 512 channels

        # Auxiliary output 1
        aux1 = None
        if self.training:
            aux1 = self.aux1(out)

        out = self.inception4b(out)     # 512 channels
        out = self.inception4c(out)     # 512 channels
        out = self.inception4d(out)     # 528 channels

        # Auxiliary output 2
        aux2 = None
        if self.training:
            aux2 = self.aux2(out)

        out = self.inception4e(out)     # 832 channels
        out = self.maxpool4(out)    # 7x7 or 1x1
        out = self.inception5a(out)     # 832 channels
        out = self.inception5b(out)     # 1024 channels
        out = self.avgpool(out)     # 1x1x1024

        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return GoogLeNetOutput(out, aux1, aux2)

    @classmethod
    def from_args(cls, args):
        dataset = args.dataset.lower()
        if dataset == 'mnist':
            return GoogLeNet(in_channels=1, out_features=10)
        elif dataset == 'cifar10':
            return GoogLeNet(in_channels=3, out_features=10)
        elif dataset == 'cifar100':
            return GoogLeNet(in_channels=3, out_features=100)
        elif dataset == 'fashionmnist':
            return GoogLeNet(in_channels=1, out_features=10)
        elif dataset == 'imagenet':
            return GoogLeNet(in_channels=3, out_features=1000)
        else:
            raise NotImplementedError('GoogLeNet not implemented for ' + args.dataset + ' dataset')