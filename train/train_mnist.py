import utils.arg_utils as utils
from tqdm import tqdm, trange
import argparse
import torch


def get_accuracy(output, target, size):
    classification = torch.argmax(output, dim=1)
    correct = (classification == target).sum().item()
    return 0


def train(model, optimizer, criterion, train_loader, device):
    model.train()
    pbar = tqdm(train_loader)
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        acc = get_accuracy(output, target, target.size(0))


def main():
    utils.seed_everything(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = utils.get_model_from_args(args).to(device)
    optimizer = utils.get_optimizer_from_args(args, model)
    criterion = torch.nn.CrossEntropyLoss()

    # Load data
    train_data, test_data = utils.get_datasets_from_args(args)
    train_loader, val_loader = utils.get_train_val_split(args, train_data)

    for epoch in trange(args.n_epochs, desc='Training', unit='epoch'):
        train(model, optimizer, criterion, train_loader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST on a image classification model')
    # Config parameters
    parser.add_argument('--model', type=str, default='MLP', metavar='M',
                        choices=['MLP', 'AlexNet', 'VGG16', 'ResNet', 'GoogLeNet'],
                        help='which model to use')
    parser.add_argument('--optim', type=str, default='SGD', metavar='O',
                        choices=['SGD', 'Adam', 'AdamW'],
                        help='which optimizer to use')
    # Data parameters
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='D',
                        choices=['MNIST'],
                        help='which dataset to use')
    parser.add_argument('--dir', type=str, default='data', metavar='D',
                        help='root data directory')
    parser.add_argument('--batch_size_train', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=5, metavar='N',
                        help='number of training epochs (default: 5)')
    parser.add_argument('--val_pc', default=0.1, type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--workers', type=int, default=0, metavar='N',
                        help='number of workers for dataloaders (default: 0)')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='N',
                        help='momentum (default: 0.9)')
    # Architecture
    parser.add_argument('--n_hidden', type=int, default=100, metavar='NH',
                        help='number of hidden layers in MLP (default: 100)')
    parser.add_argument('--n_outputs', type=int, default=10, metavar='NO',
                        help='number of output units in MLP (default: 10)')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')

    args = parser.parse_args()
    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)
    main()