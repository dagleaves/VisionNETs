from sklearn import metrics
import torch


def calc_metrics(prediction, target):
    """
    Calculate batch metrics for logging
    :param prediction: model output
    :param target: target for data
    :return: dict containing desired metrics
    """
    pred_label = torch.argmax(prediction, dim=1)
    accuracy = metrics.accuracy_score(pred_label, target)
    return {
        'accuracy': accuracy
    }


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
            save_checkpoint(model, optimizer, criterion, epoch, args)


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