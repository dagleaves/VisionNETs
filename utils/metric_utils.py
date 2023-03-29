from sklearn import metrics
import numpy as np
import torch


def calc_metrics(args, prediction, target):
    """
    Calculate batch metrics for logging
    :param args: arguments with dataset selection
    :param prediction: model output
    :param target: target for data
    :return: dict containing desired metrics
    """
    pred_label = torch.argmax(prediction, dim=1)
    top1_accuracy = metrics.accuracy_score(target, pred_label)
    if args.dataset.lower() == 'imagenet':
        with torch.no_grad():
            top5_accuracy = metrics.top_k_accuracy_score(target, prediction, k=5, labels=range(1000))
        return {
            'top1': top1_accuracy,
            'top5': top5_accuracy
        }
    return {
        'top1': top1_accuracy
    }


def update_metrics(new_metrics: dict,
                   top1_accuracy: 'AverageMeter',
                   top5_accuracy: 'AverageMeter',
                   loss: 'AverageMeter',
                   batch_size: int
                   ):
    """
    Update metric trackers with new metrics for a batch of data
    :param new_metrics:     batch metric values
    :param top1_accuracy:   top1 accuracy AverageMeter
    :param top5_accuracy:   top5 accuracy AverageMeter
    :param loss:            loss AverageMeter
    :param batch_size:      batch size
    :return: Updated tqdm postfix
    """
    top1_accuracy.update(new_metrics['top1'], batch_size)
    loss.update(new_metrics['loss'], batch_size)
    postfix = {
        'loss': '{loss.avg:.3f}'.format(loss=loss),
        'acc': '{acc.avg:.3f}'.format(acc=top1_accuracy)
    }
    if 'top5' in new_metrics.keys():
        top5_accuracy.update(new_metrics['top5'], batch_size)
        postfix['top5'] = '{top5.avg:.3f}'.format(top5=top5_accuracy)

    return postfix


def save_checkpoint(model, optimizer, criterion, epoch, args, best=False):
    """
    Save a PyTorch model checkpoint
    :param model: model to save
    :param optimizer: model's current optimizer
    :param criterion: criterion used
    :param epoch: current epoch in training
    :param args: command line arguments for model
    :param best: if model has best val loss (default False)
    :return: None
    """
    name_tail = 'best' if best else epoch + 1
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'criterion': criterion,
        'optimizer': optimizer.state_dict(),
        'args': vars(args)
    }, f'{args.ckpt_dir}/{args.model}_{args.dataset}_{name_tail}.pt')


class BestCheckpointSaver:
    """
    Class to save the best model during training based on validation loss
    """
    def __init__(self, best_loss=float('inf')):
        self.best_loss = best_loss

    def __call__(self, current_loss, model, optimizer, criterion, epoch, args):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            save_checkpoint(model, optimizer, criterion, epoch, args, best=True)


# https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/imagenet/main.py#L420
class AverageMeter:
    """
    Computes and stores an average over a series of updates
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, size=1):
        self.val = val
        self.sum += val * size
        self.count += size
        self.avg = self.sum / self.count


# https://github.com/Bjarten/early-stopping-pytorch/blob/2709576912e1f571dd60d71d0b696e25212b43ab/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), self.path)
        else:
            torch.save(model.state_dict(), self.path)

        self.val_loss_min = val_loss
