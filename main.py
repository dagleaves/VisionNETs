import utils as arg_utils
from utils import calc_metrics
from models import GoogLeNetOutput, gnet_loss
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.strategies import DDPStrategy
import argparse
import torch
import os


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = arg_utils.get_model_from_args(args)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.args = args

    def training_step(self, batch, batch_idx):
        x, target = batch
        output = self.model(x)

        if isinstance(output, GoogLeNetOutput):
            loss = gnet_loss(output, target, self.criterion)
            output = output.logits
        else:
            loss = self.criterion(output, target)

        metrics = calc_metrics(args, output.cpu(), target.cpu())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_acc", metrics['top1'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        output = self.model(x)

        if isinstance(output, GoogLeNetOutput):
            loss = gnet_loss(output, target, self.criterion)
            output = output.logits
        else:
            loss = self.criterion(output, target)

        metrics = calc_metrics(args, output.cpu(), target.cpu())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_acc", metrics['top1'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, target = batch
        output = self.model(x)

        if isinstance(output, GoogLeNetOutput):
            loss = gnet_loss(output, target, self.criterion)
            output = output.logits
        else:
            loss = self.criterion(output, target)

        metrics = calc_metrics(args, output.cpu(), target.cpu())
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("test_acc", metrics['top1'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = arg_utils.get_optimizer_from_args(args, self.model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [scheduler]


class LitDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        train_data, test_data = arg_utils.get_datasets_from_args(args)
        train_loader, val_loader = arg_utils.get_train_val_split(args, train_data)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = torch.utils.data.DataLoader(test_data,
                                                       batch_size=args.batch_size,
                                                       num_workers=args.workers,
                                                       shuffle=False)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


def main():
    global args

    pl.seed_everything(args.seed, workers=True)

    # Load data
    data = LitDataModule(args)

    # Initialize model
    model = LitModel(args)

    # Early stopping callback
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', mode='min', patience=3)

    # Initialize trainer
    if torch.cuda.is_available():
        ddp_backend = 'gloo' if os.name == 'nt' else 'nccl'     # windows does not support nccl
        ddp = DDPStrategy(process_group_backend=ddp_backend)
        trainer = pl.Trainer(default_root_dir=args.ckpt_dir,
                             callbacks=early_stop_callback,
                             max_epochs=args.epochs,
                             accelerator='gpu',
                             devices=args.devices,
                             num_nodes=args.nodes,
                             strategy=ddp
                             )
    else:
        trainer = pl.Trainer(default_root_dir=args.ckpt_dir, callbacks=early_stop_callback, max_epochs=args.epochs)

    trainer.fit(model, data)
    trainer.test(model, data)


def cli_main():
    cli = LightningCLI(LitModel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train common image classification models')
    # Config parameters
    parser.add_argument('--model', type=str, default='MLP', metavar='M',
                        choices=['MLP', 'AlexNet', 'LeNet5', 'VGG16', 'ResNet50', 'GoogLeNet'],
                        help='which model to use')
    parser.add_argument('--optim', type=str, default='SGD', metavar='O',
                        choices=['SGD', 'Adam', 'AdamW'],
                        help='which optimizer to use')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', metavar='C',
                        help='saved model checkpoints directory')
    # Data parameters
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='D',
                        choices=['MNIST', 'CIFAR10', 'CIFAR100', 'FashionMNIST', 'ImageNet'],
                        help='which dataset to use')
    parser.add_argument('--data_dir', type=str, default='data', metavar='D',
                        help='root data directory')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of training epochs (default: 1)')
    parser.add_argument('--val_pc', default=0.1, type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--workers', type=int, default=0, metavar='N',
                        help='number of workers for dataloaders (default: 0)')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='N',
                        help='momentum (default: 0.9)')
    parser.add_argument('--decay', type=float, default=0.0, metavar='N',
                        help='weight decay (default: 0.0)')
    # Distributed training parameters
    parser.add_argument('--devices', type=int, default=1, metavar='N',
                        help='number of gpus per node (default: 1)')
    parser.add_argument('--nodes', type=int, default=1, metavar='N',
                        help='number of nodes for training (default: 1)')
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')

    # Load args
    args = parser.parse_args()

    # Print args
    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)

    # Train model
    main()
