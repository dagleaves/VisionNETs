import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, in_features: int = 28**2, hidden_features: int = 100, n_hidden: int = 1, out_features: int = 10):
        """
        Initialize a basic multi-layer perceptron (MLP)
        :param in_features: number of input features. Default 784 for MNIST 28x28 images
        :param hidden_features: size (width) of hidden layer(s). Default 100
        :param n_hidden: number of hidden layers to use. Default 1
        :param out_features: number of classes. Default 10 MNIST
        """
        super(MLP, self).__init__()
        self.inp_lin = nn.Linear(in_features, hidden_features, bias=False)
        self.hidden = nn.ModuleList([nn.Linear(hidden_features, hidden_features, bias=False) for _ in range(n_hidden)])
        self.out_fc = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = F.relu(self.inp_lin(x))
        for hidden in self.hidden:
            x = F.relu(hidden(x))
        x = self.out_fc(x)
        return x
