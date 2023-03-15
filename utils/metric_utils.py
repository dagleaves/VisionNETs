from sklearn import metrics


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