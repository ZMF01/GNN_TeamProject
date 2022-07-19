"""
This version uses the cross entropy loss function of pytorch, which consists of log_softmax and NLL
"""
import torch
import torch.nn.functional as F
import random
import argparse
from torch.utils.data import DataLoader
import utils
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score
import math

from load import load_data
import torch as th
from data_split import split_dataset
from utils import EarlyStopping

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def my_f1_score(labels, pred, average='micro'):
    true = labels.data.cpu().numpy()
    pred = pred.data.cpu().numpy()
    # pred = np.argmax(pred[mask].data.cpu.numpy(), axis=1)
    score = f1_score(true, pred, average=average)
    return score


def evaluate(feats, labels, model, loss_fcn):
    with th.no_grad():
        model.eval()
        output = model(feats)
        predict = th.argmax(output, dim=1).data.cpu().numpy()
        loss = loss_fcn(output, labels)

        labels = labels.cpu().numpy()

        score = f1_score(labels, predict, average='micro')

        return score, loss.item()


def train_main(args):
    check_point_path = "./es_checkpoint_logistic.pt"
    split_path = "./split.pt"
    # load data
    G, feats, labels, input_dim, num_classes = \
        load_data(args.dataset, True)

    if args.not_fix_split:
        train_mask, val_mask, test_mask, _, _, _ = split_dataset(
        args.dataset,
        args.train_percent,
        args.val_percent)
        th.save([train_mask, val_mask, test_mask], split_path)
    else:
        train_mask, val_mask, test_mask = th.load(split_path)

    # cpu or gpu
    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))

    model = nn.Linear(input_dim, num_classes)
    feats = feats.to(device)
    labels = labels.to(device)
    model = model.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # Loss and optimizer
    loss_func = nn.CrossEntropyLoss()  # computes softmax internally
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train the model
    print("Training begins: num_epoch=%d" % (args.epochs))
    early_stopper = EarlyStopping(check_point_path, '<', args.patience)
    for epoch in range(args.epochs):
        model.train()

        train_logits = model(feats[train_mask])
        train_pred = train_logits.argmax(dim=1)
        train_loss = loss_func(train_logits, labels[train_mask])
        train_f1_score = my_f1_score(labels[train_mask], train_pred, average='micro')

        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        val_score, val_loss = evaluate(feats[val_mask], labels[val_mask], model, loss_func)
        test_score, test_loss = evaluate(feats[test_mask], labels[test_mask], model, loss_func)

        print("Epoch {:05d} | Train Loss {:.4f} | Train F1-score {:.4f} | Val Loss {:.4f} | Val F1-score {:.6f} | "
              "Test Loss {:.4f} | Test F1-score {:.6f}".format(
            epoch, train_loss.item(), train_f1_score, val_loss, val_score, test_loss, test_score))

        if early_stopper.step(val_score, model):
            break

    model.load_state_dict(th.load(check_point_path))
    model.eval()
    test_score, test_loss = evaluate(feats, labels, model, loss_func)

    print("Test Loss {:.4f} | Test F1-score {:.8f} on {}".format(test_loss, test_score, args.dataset))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='logistic_regression')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.02,
                        help="learning rate")
    parser.add_argument('--patience', type=int, default=100,
                        help="used for early stop")
    parser.add_argument('--dataset', type=str, default='cora',
                        help="which dataset to use")
    parser.add_argument('--directed', type=bool, default=False,
                        help="whether directed")
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--not-fix-split", action='store_true')
    parser.add_argument("--train_percent", type=float, default=0.05)
    parser.add_argument("--val_percent", type=float, default=0.2)
    args = parser.parse_args()
    print(args)
    train_main(args)