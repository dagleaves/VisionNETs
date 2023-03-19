from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.optim import SGD, Adam, AdamW
from pathlib import Path
from models import MLP, LeNet5, AlexNet
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
    elif model_arg == 'alexnet':
        return AlexNet.from_args(args)
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


def get_dataset_mean_std(dataset):
    """
    Get the mean and std of a dataset for normalization
    :param dataset: dataset name
    :return: mean, std
    """
    if dataset == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
    elif dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset == 'cifar100':
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
    else:
        raise NotImplementedError('Mean and std not available for ' + dataset + ' dataset')
    return mean, std


def get_resize_transforms(dataset, model):
    # Non-imagenet dataset
    if dataset != 'imagenet':
        if model in ['mlp', 'lenet5']:  # No resizing necessary
            return []
        else:   # upscale to 64x64
            return [
                transforms.Resize((70, 70)),
                transforms.CenterCrop((64, 64))
            ]
    if model in ['mlp', 'lenet5']:
        raise NotImplementedError('Can ImageNet be used for MLP or LeNet5? # TODO')
    else:   # Standard ImageNet size
        return [
            transforms.Resize((227, 227)),
            transforms.CenterCrop((224, 224))
        ]


def get_transforms_from_args(args):
    """
    Get dataset transforms from args
    :param args: dataset and model selection
    :return: torchvision.transforms.Compose
    """
    dataset = args.dataset.lower()
    model = args.model.lower()
    mean, std = get_dataset_mean_std(dataset)

    # Compose transforms
    tfs = get_resize_transforms(dataset, model)
    tfs += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if model == 'mlp':
        tfs.append(transforms.Lambda(lambda x: torch.flatten(x)))
    return transforms.Compose(tfs)


def get_datasets_from_args(args):
    """
    Load dataset from args for given dataset name
    :param args: args with dataset name
    :return: [train, test] datasets
    """
    dataset = args.dataset.lower()
    tfs = get_transforms_from_args(args)
    if dataset == 'mnist':
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
    elif dataset == 'cifar100':
        # Load datasets
        train_data = datasets.CIFAR100(args.data_dir,
                                       train=True,
                                       download=True,
                                       transform=tfs
                                       )
        test_data = datasets.CIFAR100(args.data_dir,
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
