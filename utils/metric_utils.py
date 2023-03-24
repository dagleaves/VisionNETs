from sklearn import metrics
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
