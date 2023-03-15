from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.optim import SGD, Adam, AdamW
from models.MLP import MLP
import numpy as np
import random
import torch
import os


def get_model_from_args(args):
    """
    Load model from specified arg
    :param args: args with args.model specifiying model to use
    :return: model object: torch.nn.Module
    """
    model_arg = args.model.lower()
    if model_arg == 'MLP':
        return MLP()
    else:
        raise NotImplemented('Model choice does not match an implemented option')


def get_optimizer_from_args(args, model):
    """
    Load optimizer from args for given model parameters
    :param args: args with optimizer name
    :param model: torch.nn.Module with parameters to optimize
    :return: pytorch optimizer
    """
    optim_arg = args.optim.lower()
    if optim_arg == 'sgd':
        return SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif optim_arg == 'adam':
        return Adam(model.parameters(), lr=args.lr)
    elif optim_arg == 'adamw':
        return AdamW(model.parameters(), lr=args.lr)
    else:
        raise NotImplemented('Optimizer does not match an implemented option')


def get_datasets_from_args(args):
    """
    Load dataset from args for given dataset name
    :param args: args with dataset name
    :return: [train, test] datasets
    """
    dataset = args.dataset.lower()
    if dataset == 'mnist':
        mean = (0.1307,)    # magic MNIST mean
        std = (0.3081,)     # magic MNIST std
        tfs = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean, std)])
        train_data = datasets.MNIST(args.dir,
                                    train=True,
                                    download=True,
                                    transform=tfs
                                    )
        test_data = datasets.MNIST(args.dir,
                                   train=False,
                                   download=True,
                                   transform=tfs)
        return train_data, test_data
    else:
        raise NotImplemented('Dataset does not match an implemented option')


def get_train_val_split(args, dataset):
    """
    Split dataset into train and val splits, according to provided arg percent
    :param args: args with validation split percentage
    :param dataset: dataset to split
    :return: [train, validation] dataloaders
    """
    val_split = int(len(dataset) * args.val_pc)
    train_sampler = SubsetRandomSampler(list(range(val_split)))
    val_sampler = SubsetRandomSampler(list(range(val_split, len(dataset))))
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size_train,
                                               sampler=train_sampler,
                                               num_workers=args.workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size_test,
                                             sampler=val_sampler,
                                             num_workers=args.workers,
                                             shuffle=True)
    return train_loader, val_loader


def seed_everything(args):
    """
    Set seeds for reproducibility
    :param args: args with specified seed
    :return: None
    """
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
