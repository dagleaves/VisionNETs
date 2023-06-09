from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.optim import SGD, Adam, AdamW
from pathlib import Path
from models import MLP, LeNet5, AlexNet, VGG16, ResNet50, GoogLeNet
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
    elif model_arg == 'vgg16':
        return VGG16.from_args(args)
    elif model_arg == 'resnet50':
        return ResNet50.from_args(args)
    elif model_arg == 'googlenet':
        return GoogLeNet.from_args(args)
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
        return SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
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
    elif dataset == 'fashionmnist':
        mean = (0.2860,)
        std = (0.3530,)
    elif dataset == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise NotImplementedError('Mean and std not available for ' + dataset + ' dataset')
    return mean, std


def get_resize_transforms(model):
    if model == 'alexnet':
        return [
            transforms.Resize((227, 227)),
            transforms.CenterCrop((224, 224))
        ]
    return []


def get_transforms_from_args(args):
    """
    Get dataset transforms from args
    :param args: dataset and model selection
    :return: torchvision.transforms.Compose
    """
    dataset = args.dataset.lower()
    model = args.model.lower()
    mean, std = get_dataset_mean_std(dataset)

    tfs = get_resize_transforms(model)
    tfs += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(tfs)


def get_datasets_from_args(args):
    """
    Load dataset from args for given dataset name
    :param args: args with dataset name
    :return: [train, test] datasets
    """
    dataset = args.dataset.lower()
    tfs = get_transforms_from_args(args)
    # Load datasets
    if dataset == 'mnist':
        train_data = datasets.MNIST(args.data_dir,
                                    train=True,
                                    download=True,
                                    transform=tfs
                                    )
        test_data = datasets.MNIST(args.data_dir,
                                   train=False,
                                   download=True,
                                   transform=tfs
                                   )
    elif dataset == 'cifar10':
        train_data = datasets.CIFAR10(args.data_dir,
                                      train=True,
                                      download=True,
                                      transform=tfs
                                      )
        test_data = datasets.CIFAR10(args.data_dir,
                                     train=False,
                                     download=True,
                                     transform=tfs
                                     )
    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(args.data_dir,
                                       train=True,
                                       download=True,
                                       transform=tfs
                                       )
        test_data = datasets.CIFAR100(args.data_dir,
                                      train=False,
                                      download=True,
                                      transform=tfs
                                      )
    elif dataset == 'fashionmnist':
        train_data = datasets.FashionMNIST(args.data_dir,
                                           train=True,
                                           download=True,
                                           transform=tfs
                                           )
        test_data = datasets.FashionMNIST(args.data_dir,
                                          train=False,
                                          download=True,
                                          transform=tfs
                                          )
    elif dataset == 'imagenet':
        train_data = datasets.ImageNet(args.data_dir + '/imagenet',
                                       split='train',
                                       transform=tfs
                                       )
        test_data = datasets.ImageNet(args.data_dir + '/imagenet',
                                      split='val',
                                      transform=tfs
                                      )
    else:
        raise NotImplementedError('Dataset ' + args.dataset + ' not implemented')
    return train_data, test_data


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
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True
                                               )
    val_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             sampler=val_sampler,
                                             num_workers=args.workers,
                                             pin_memory=True
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
    source_dir = Path(__file__).resolve().parent    # utils folder
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
