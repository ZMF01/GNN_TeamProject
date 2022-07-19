# coding=utf-8
import hashlib
import operator
import os
import time
import numpy as np
import networkx as nx
import torch as th
import matplotlib as mlt

# mlt.use('Qt5Agg')
from metrics import my_f1_score

mlt.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd


def evaluate(G, feats, labels, mask, model, loss_fcn, feats_t=None):
    with th.no_grad():
        model.eval()
        output = model(G, feats)
        predict = th.argmax(output, dim=1)
        loss = loss_fcn(output[mask], labels[mask])
        score = my_f1_score(labels, predict, mask)

        # true = np.argmax(labels[mask].data.cpu().numpy(), axis=1)
        # score = f1_score(true, predict, average='micro')
        return score, loss.item()


def convert_multitarget2_class_index(n):
    """
    :param n: ndarray or Tensor
    :return:
    """
    if isinstance(n, np.ndarray):
        m = np.argmax(n, axis=1)
    elif isinstance(n, th.Tensor):
        m = th.argmax(n, dim=1)
    else:
        assert False, "wrong input type"
    return m


def plot_res(x, y1, y2, labels=['train', 'val'], intv=0.1, x_axis=None, x_label='alpha',
             y_label='f1 score', location='lower left'):
    """
    draw scatter plot
    :param x:
    :param y1:
    :param y2:
    :param labels:
    :param intv:
    :param x_axis:
    :param x_label:
    :param y_label:
    :param location:
    :return:
    """
    ax = plt.gca()
    plt.plot(x, y1, color='red', marker='o', linestyle='-', label=labels[0])
    plt.plot(x, y2, color='blue', marker='^', linestyle='-', label=labels[1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x_axis is None:
        ax.xaxis.set_major_locator(plt.MultipleLocator(intv))
    else:
        ax.set_xticks(x_axis)
    plt.grid(True)
    plt.legend(loc=location)
    plt.pause(0.1)


def show_statistics(labels_index, train_list, val_list, test_list):
    """
    after splitting the dataset into 3 sets, show some statistics, including in each subset,
    the proportionn of each class
    :param labels_index:
    :param train_list:
    :param val_list:
    :param test_list:
    :return:
    """
    # show some statistics for partitioned data
    # data = labels_index.argmax(dim=1, keepdim=True).cpu().data.numpy()
    if isinstance(labels_index, th.Tensor):
        labels_index = labels_index.cpu().data.numpy()

    train_labels = pd.DataFrame(labels_index[train_list])
    val_labels = pd.DataFrame(labels_index[val_list])
    test_labels = pd.DataFrame(labels_index[test_list])

    id = list(train_labels.columns)[-1]
    print("In training set:\n", train_labels[id].value_counts() / len(train_labels))
    print("\nIn val set:\n", val_labels[id].value_counts() / len(val_labels))
    print("\nIn test set:\n", test_labels[id].value_counts() / len(test_labels))
    print(f"\nThere are {labels_index.shape[0]} examples")
    print(f"\nTraining Set: {len(train_labels)} | Val Set: {len(val_labels)} | Test Set: {len(test_labels)} "
          f"| In Total: {len(train_labels) + len(val_labels) + len(test_labels)}")


def draw_nx_graph(G):
    """
    visual networkx graph
    :param G:
    :return:
    """
    # nx.draw_shell(G)
    # nx.draw_spectral(G)
    nx.draw_circular(G, with_labels=True)
    # nx.draw_spring(G)
    # nx.draw_random(G)


class EarlyStopping:
    def __init__(self, save_path, op: str, patience: int = 100):
        """
        :param op: min or max
        :param patience:
        """
        if op == '<':
            self.cmp = operator.lt
        elif op == '>':
            self.cmp = operator.gt
        else:
            print("Please provide valid param op for EarlyStopping", op)
            assert False
        self.patience = patience  # maximum no. of iterations to wait for without improvement
        self.counter = 0  # how many iterations have waited
        self.best_score = None
        self.early_stop = False
        self._save_path = save_path

    def step(self, acc, model):
        """
        acc: accuracy of the current iteration
        model:
        """
        score = acc
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
        elif self.cmp(score, self.best_score):  # < for accuracy and > for loss
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def _save_checkpoint(self, model):
        """
        Saves model when validation loss decrease.
        """
        th.save(model.state_dict(), self._save_path)
        print('es_checkpoint.pt saved')


def get_checkpoint_path(name):
    time_stamp = str(time.time())
    print(time_stamp)

    hash_str = hashlib.md5()
    hash_str.update(time_stamp.encode())
    md5_test = hash_str.hexdigest()
    # print(md5_test)  # 打印加密后的md5值
    name += md5_test
    return name


def compute_dataset_folder(dataset):
    return os.path.join('.', 'datasets', dataset)


def compute_paths(dataset):
    folder = compute_dataset_folder(dataset)
    G_path = os.path.join(folder, f"{dataset}-G.json")
    feats_path = os.path.join(folder, f"{dataset}-feats.npy")
    class_map_path = os.path.join(folder, f"{dataset}-class_map.json")
    return G_path, feats_path, class_map_path
