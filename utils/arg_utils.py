from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.optim import SGD, Adam, AdamW
from pathlib import Path
from models import MLP, LeNet5
import numpy as np
import random
import torch
import wandb
import yaml
import os


def get_model_from_args(args):
    """
    Load model from specified arg
    :param args: args with args.model specifiying model to use
    :return: model object: torch.nn.Module
    """
    model_arg = args.model.lower()
    if model_arg == 'mlp':
        return MLP.from_args(args)
    elif model_arg == 'lenet5':
        return LeNet5.from_args(args)
    else:
        raise NotImplementedError(f'Model {args.model} is not implemented')


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
        raise NotImplementedError(f'Optimizer {args.optim} is not implemented')


def get_datasets_from_args(args):
    """
    Load dataset from args for given dataset name
    :param args: args with dataset name
    :return: [train, test] datasets
    """
    dataset = args.dataset.lower()
    if dataset == 'mnist':
        # Define transforms
        mean = (0.1307,)    # magic MNIST mean
        std = (0.3081,)     # magic MNIST std
        tfs = [transforms.ToTensor(),
               transforms.Normalize(mean, std),
               ]
        if args.model.lower() == 'mlp':
            tfs.append(transforms.Lambda(lambda x: torch.flatten(x)))
        tfs = transforms.Compose(tfs)

        # Load datasets
        train_data = datasets.MNIST(args.data_dir,
                                    train=True,
                                    download=True,
                                    transform=tfs
                                    )
        test_data = datasets.MNIST(args.data_dir,
                                   train=False,
                                   download=True,
                                   transform=tfs)
        return train_data, test_data
    elif dataset == 'cifar10':
        # Define transforms
        mean = (0.4914, 0.4822, 0.4465,)    # magic MNIST mean
        std = (0.2470, 0.2435, 0.2616,)     # magic MNIST std
        tfs = [transforms.ToTensor(),
               transforms.Normalize(mean, std),
               ]
        if args.model.lower() == 'mlp':
            tfs.append(transforms.Lambda(lambda x: torch.flatten(x)))
        tfs = transforms.Compose(tfs)

        # Load datasets
        train_data = datasets.CIFAR10(args.data_dir,
                                      train=True,
                                      download=True,
                                      transform=tfs
                                      )
        test_data = datasets.CIFAR10(args.data_dir,
                                     train=False,
                                     download=True,
                                     transform=tfs)
        return train_data, test_data
    else:
        raise NotImplementedError(f'Dataset {args.dataset} in not implemented')


def get_train_val_split(args, dataset):
    """
    Split dataset into train and val splits, according to provided arg percent
    :param args: args with validation split percentage
    :param dataset: dataset to split
    :return: [train, validation] dataloaders
    """
    val_split = int(len(dataset) * args.val_pc)
    val_sampler = SubsetRandomSampler(list(range(val_split)))
    train_sampler = SubsetRandomSampler(list(range(val_split, len(dataset))))
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size_train,
                                               sampler=train_sampler,
                                               num_workers=args.workers,
                                               )
    val_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size_test,
                                             sampler=val_sampler,
                                             num_workers=args.workers
                                             )
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


def setup_sweep(args):
    """
    Setup WandB hyperparameter sweep
    :param args: dict to assign hyperparameters to
    :return: sweep_id
    """
    if not args.sweep:
        return None
    assert args.wandb, 'In order to sweep, wandb must be enabled'

    # Load sweep config
    source_dir = Path(__file__).resolve().parent
    with open(source_dir / 'wandb_config.yaml', 'r') as config_file:
        sweep_config = yaml.safe_load(config_file)

    sweep_id = wandb.sweep(sweep_config, project='VisionNETs-sweep')
    return sweep_id


def get_sweep_args(args):
    """
    Get wandb sweep config params
    :param args: current args
    :return: updated args
    """
    if not args.wandb or not args.sweep:
        return args
    # Set hyperparameters from config
    args.lr = wandb.config.learning_rate
    args.batch_size_train = wandb.config.batch_size
    args.n_epochs = wandb.config.epochs
    args.optim = wandb.config.optimizer
    return args


def init_wandb(args):
    """
    Initizalize wandb run according to args
    :param args: if wandb enabled and if sweep
    :return: None
    """
    if not args.wandb:
        return
    if args.sweep:
        wandb.init()
    else:
        wandb.init(project='VisionNETs')


def wandb_log(args, log):
    """
    Log metrics to wandb
    :param args: if wandb is used
    :param log: metrics to log
    :return: None
    """
    if not args.wandb:
        return
    wandb.log(log)
