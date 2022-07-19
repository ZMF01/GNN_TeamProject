# coding=utf-8
import math
import operator
import networkx as nx
import torch
import numpy as np
from enum import Enum
import random
from load import loadG, loadndarray, loadjson, process_class_map
from utils import convert_multitarget2_class_index, show_statistics, compute_dataset_folder, compute_paths
import os


def split_dataset(dataset: str, train_percent, val_percent):
    """
    split the dataset into train, val, test set
    :param val_percent:
    :param train_percent:
    :param folder:
    :return: train_mask, val_mask, test_mask
    """
    G_path, _, class_map_path = compute_paths(dataset)
    G = loadG(G_path)
    class_map = loadjson(class_map_path)
    labels = process_class_map(G, class_map)
    labels = convert_multitarget2_class_index(labels)

    num_train = int(labels.shape[0] * train_percent)
    num_val = int(labels.shape[0] * val_percent)

    shuffle_index = list(range(labels.shape[0]))
    random.shuffle(shuffle_index)

    train_list = shuffle_index[:num_train]
    val_list = shuffle_index[num_train:num_train + num_val]
    test_list = shuffle_index[num_train + num_val:]

    show_statistics(labels, train_list, val_list, test_list)

    train_mask, val_mask, test_mask = np.zeros((len(labels),)), \
                                      np.zeros((len(labels),)), np.zeros((len(labels),))
    train_mask[train_list] = 1
    val_mask[val_list] = 1
    test_mask[test_list] = 1
    train_mask, val_mask, test_mask = torch.BoolTensor(train_mask), torch.BoolTensor(val_mask), torch.BoolTensor(
        test_mask)

    return train_mask, val_mask, test_mask, train_list, val_list, test_list


# def split_dataset_ids(folder: str, train_percent):
#     """
#     split the dataset into train, val, test set
#     :param val_percent:
#     :param train_percent:
#     :param folder:
#     :return: train_mask, val_mask, test_mask
#     """
#     # class_map = loadjson(f"{os.environ['PROJECT_HOME']}/data/{dataset}/class_map.json")
#     G = loadG(os.path.join(folder, "G.json"))
#     class_map = loadjson(os.path.join(folder, "class_map.json"))
#     labels = process_class_map(G, class_map)
#     labels = torch.FloatTensor(labels)
#
#     train_mask, test_mask = np.zeros((len(labels),)), np.zeros((len(labels),))
#
#     shuffle_index = list(range(labels.shape[0]))
#     random.shuffle(shuffle_index)
#
#     num_train = int(labels.shape[0] * train_percent)
#
#     train_list = shuffle_index[:num_train]
#     test_list = shuffle_index[num_train:]
#
#     show_statistics(labels, train_list, test_list)
#
#     return train_list, test_list
