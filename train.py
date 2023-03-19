import utils as arg_utils
from utils import AverageMeter, calc_metrics, BestCheckpointSaver, save_checkpoint
from tqdm import tqdm, trange
import argparse
import torch
import wandb


def train(model, optimizer, criterion, train_loader, device, epoch):
    train_accuracy = AverageMeter()
    train_loss = AverageMeter()
    model.train()

    pbar = tqdm(train_loader, desc='Epoch ' + str(epoch + 1), unit='batch', leave=True)
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Update model parameters
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update metrics
        metrics = calc_metrics(output.cpu(), target.cpu())
        train_accuracy.update(metrics['accuracy'], target.size(0))
        train_loss.update(loss.item(), target.size(0))
        pbar.set_postfix({
            'loss': '{loss.val:.3f} ({loss.avg:.3f})'.format(loss=train_loss),
            'acc': '{acc.val:.3f} ({acc.avg:.3f})'.format(acc=train_accuracy)
        })
        arg_utils.wandb_log(args, {
            'epoch': epoch + 1,
            'train_loss': train_loss.avg,
            'train_acc': train_accuracy.avg
        })


def test(model, criterion, data_loader, device, testing=False):
    test_accuracy = AverageMeter()
    test_loss = AverageMeter()
    model.eval()

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Validation' if not testing else 'Testing', unit='batch', leave=True)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            # Update metrics
            metrics = calc_metrics(output.cpu(), target.cpu())
            test_accuracy.update(metrics['accuracy'], target.size(0))
            test_loss.update(loss.item(), target.size(0))
            pbar.set_postfix({
                'loss': '{loss.val:.3f} ({loss.avg:.3f})'.format(loss=test_loss),
                'acc': '{acc.val:.3f} ({acc.avg:.3f})'.format(acc=test_accuracy)
            })
            arg_utils.wandb_log(args, {
                'val_loss': test_loss.avg,
                'val_acc': test_accuracy.avg
            })

    return test_loss.avg, test_accuracy.avg


def main():
    global args
    arg_utils.init_wandb(args)
    arg_utils.seed_everything(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Get updated sweep args if enabled
    args = arg_utils.get_sweep_args(args)

    model = arg_utils.get_model_from_args(args).to(device)
    optimizer = arg_utils.get_optimizer_from_args(args, model)
    criterion = torch.nn.CrossEntropyLoss()
    save_best_checkpoint = BestCheckpointSaver()

    # Load data
    train_data, test_data = arg_utils.get_datasets_from_args(args)
    train_loader, val_loader = arg_utils.get_train_val_split(args, train_data)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size_test,
                                              num_workers=args.workers,
                                              shuffle=False)

    # Train model
    pbar = trange(args.n_epochs, desc='Training', unit='epoch')
    for epoch in pbar:
        train(model, optimizer, criterion, train_loader, device, epoch)
        val_loss, val_acc = test(model, criterion, val_loader, device)
        save_best_checkpoint(val_loss, model, criterion, optimizer, epoch, args)

        # Logging
        pbar.set_postfix({
            'loss': val_loss,
            'acc': val_acc
        })

    # Save final model
    save_checkpoint(model, optimizer, criterion, epoch, args)

    # Test model on test set
    test_loss, test_acc = test(model, criterion, test_loader, device, testing=True)
    print('Test Loss:', '%.3f' % test_loss)
    print('Test Accuracy', test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST on a image classification model')
    # Config parameters
    parser.add_argument('--model', type=str, default='MLP', metavar='M',
                        choices=['MLP', 'AlexNet', 'LeNet5', 'VGG16', 'ResNet', 'GoogLeNet'],
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
    parser.add_argument('--batch_size_train', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=2, metavar='N',
                        help='number of training epochs (default: 2)')
    parser.add_argument('--val_pc', default=0.1, type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--workers', type=int, default=0, metavar='N',
                        help='number of workers for dataloaders (default: 0)')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='N',
                        help='momentum (default: 0.9)')
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')
    parser.add_argument('--wandb', action='store_true',
                        help='use wandb (default: false)')
    parser.add_argument('--sweep', action='store_true',
                        help='use WandB hyperparameter sweep (default: false)')

    # Load args
    args = parser.parse_args()
    if args.sweep and not args.wandb:
        parser.error('--sweep requires --wandb')    # ensure wandb enabled for sweep

    # Print args
    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)

    # Train model
    sweep_id = arg_utils.setup_sweep(args)
    if sweep_id is not None:
        wandb.agent(sweep_id, function=main, count=10)
    else:
        main()
