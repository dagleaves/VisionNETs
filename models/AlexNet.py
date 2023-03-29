import torch.nn as nn
import torch


class AlexNet(nn.Module):
    """
    Implementation of the AlexNet architecture
    Using https://arxiv.org/abs/1404.5997 version
    """
    def __init__(self, in_channels: int, out_features: int):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
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
            return AlexNet(in_channels=1, out_features=10)
        elif dataset == 'cifar10':
            return AlexNet(in_channels=3, out_features=10)
        elif dataset == 'cifar100':
            return AlexNet(in_channels=3, out_features=100)
        elif dataset == 'fashionmnist':
            return AlexNet(in_channels=1, out_features=10)
        elif dataset == 'imagenet':
            return AlexNet(in_channels=3, out_features=1000)
        else:
            raise NotImplementedError('AlexNet not implemented for ' + args.dataset + ' dataset')
