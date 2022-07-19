import torch
from sklearn.metrics import f1_score


def my_f1_score(labels, pred, mask, average='micro'):
    if isinstance(labels, torch.Tensor):
        if len(labels.shape) != 1:
            labels = torch.argmax(labels, dim=1)
        if labels.is_cuda:
            true = labels[mask].data.cpu()
        else:
            true = labels[mask].data
    else:
        pass

    true = true.numpy()
    # pred = pred[mask].data.cpu()
    pred = pred[mask].data.cpu().numpy()
    # pred = np.argmax(pred[mask].data.cpu.numpy(), axis=1)
    score = f1_score(true, pred, average=average)
    return score


def evaluate_f1(model, labels: torch.Tensor, loss_fcn, mask, *input):
    """
    public method: only support computation of f1-score for single-class classification problem
    :param model:
    :param features:
    :param labels:
    :param mask:
    :return:
    """
    model.eval()
    with torch.no_grad():
        logits = model(*input)
        loss = loss_fcn(logits[mask], labels[mask])
        _, pred = torch.max(logits, dim=1)
        # convert logits and labels into index format
        score = my_f1_score(labels, pred, mask)
        return score, loss.item()


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate_accuracy(model, labels, loss_fcn, mask, *input):
    model.eval()
    with torch.no_grad():
        logits = model(*input)
        loss = loss_fcn(logits[mask], labels[mask])
        _, pred = torch.max(logits, dim=1)
        correct = torch.sum(pred == labels)
        accuracy = correct.item() * 1.0 / len(labels)
        return accuracy, loss.item()