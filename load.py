# coding=utf-8
import networkx as nx
import numpy as np
import json
import torch as th
from networkx.readwrite import json_graph
import random
from dgl import DGLGraph
import os
from utils import convert_multitarget2_class_index, compute_paths


def loadjson(x):
    """
    des: load a file in json format
    return: dict
    """
    print(f"loading {x}")
    return json.load(open(x))


def loadndarray(path):
    print(f"loading {path}")
    return np.load(path)


def loadG(path, directed=False):
    """
    load G.json and transform into a networkx Graph object
    """
    # print(f"loading {path}")
    return json_graph.node_link_graph(loadjson(path), directed)


## convert_ndarray, convert_list, convert_dict
def convert_ndarray(x):
    """
    Transform a dict into ndarray, the dict's keys are like "0", "1" ... or 0, 1, 2. ie. represent consequent integers
    example:
    >>> x={0: v1, 1:v2, 2:v3}
    >>> result = [v1,v2,v3]
    x: dict
    """
    y = [0] * len(x)
    for k, v in x.items():
        y[int(k)] = v
    return np.array(y)


def convert_list(x):
    """
    x is the class_map dict, convert x'values, ie. labels into list format
    ------
    example:
    >>> x = {0:1, 1:4}
    >>> result = {0:[1,0,0,0,0], 1:[0,0,0,0,1]}
    ------
    return: still a dict but values' format changed
    """
    c = []

    ## fetch all the values of x, and store them in c
    for k, v in x.items():
        if v not in c:
            c.append(v)  # now c contains all the possible labels

    new_x = {}
    for k, v in x.items():
        v_new = [0] * len(c)
        v_new[c.index(v)] = 1
        new_x[k] = v_new
    return new_x


def convert_dict(d, k_conversion, v_conversion=int):
    """
    convert the keys and values of a given dict according to the given conversion functions

    d: dict
    k_conversion, v_conversion: the two conversion functions for keys and values respectively
    """
    return {k_conversion(k): v_conversion(v) for k, v in d.items()}


def process_class_map(G, class_map):
    """
    convert the class_map from dict format into ndarray format
    """
    # 判断class_map的value是不是list，如果是，保持不变。如果不是，那么可能是int也可能是str，先变成int再说。下面再把int变为list
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n: n  # if is vector, keep it
    else:
        lab_conversion = lambda n: int(n)

    # 根据G的那边的id的格式，来判断需要对class_map做什么格式处理，如果G那边是int，class_map这里也要改成int，如果G那里是str，class_map这里就不需要变(因为json的key只能是str)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n: int(n)
    else:
        conversion = lambda n: n

    class_map = convert_dict(class_map, conversion, lab_conversion)  # process class_map, one for key, one for value

    # if single label, transform to vector representation
    for k, v in class_map.items():
        if type(v) != list:
            class_map = convert_list(class_map)
        break

    # Now, class_map is a dict, whose id has the same format as G and whose values are in list format
    class_map = convert_ndarray(class_map)  # 转化为了ndarray对象

    return class_map


def load_G_feats_labels(dataset="cora", directed=True):
    # load preprocessed data
    """
    the function is only to load preprocessed data, doesn't have other functionalities

    G: networkx Graph
    feats: ndarray
    class_map： dict
    """
    # 1. feats
    feats = loadndarray(f"{os.environ['PROJECT_HOME']}/data/{dataset}/feats.npy")
    # 2. class_map
    class_map = loadjson(f"{os.environ['PROJECT_HOME']}/data/{dataset}/class_map.json")
    # 3. G.json
    G = loadG(f"{os.environ['PROJECT_HOME']}/data/{dataset}/G.json", directed)

    return G, feats, class_map


def load_data(dataset="as_small", directed=True):
    """
    :param dataset:
    :param directed:
    :return: features, labels: Tensor
    """
    G_path, feats_path, class_map_path = compute_paths(dataset)
    G = loadG(G_path)  # G2 contains 2 types of edges

    n_nodes = G.number_of_nodes()
    # make sure that there exists exactly only one self-loop for each node
    G.remove_edges_from(nx.selfloop_edges(G))  # call networkx's method to remove self-loops
    G.add_edges_from([(i, i) for i in range(n_nodes)])

    feats = loadndarray(feats_path)
    labels = loadjson(class_map_path)

    feats = th.FloatTensor(feats)
    labels = process_class_map(G, labels)
    labels = th.LongTensor(labels)

    num_features = feats.shape[1]
    num_labels = labels.shape[1]

    # g = DGLGraph()
    # g.from_networkx(G)

    # print some statistics
    labels = convert_multitarget2_class_index(labels)
    return G, feats, labels, num_features, num_labels


# def load_data_multitarget(dataset="as_small", load_topo=True, directed=True):
#     """
#     :param dataset:
#     :param directed:
#     :return: features, labels: Tensor
#     """
#     G = loadG(f"{os.environ['PROJECT_HOME']}/data/{dataset}/G.json")  # G2 contains 2 types of edges
#     n_nodes = G.number_of_nodes()
#
#     G.remove_edges_from(nx.selfloop_edges(G))  # call networkx's method to remove self-loops
#     G.add_edges_from([(i, i) for i in range(n_nodes)])
#
#     feats = loadndarray(f"{os.environ['PROJECT_HOME']}/data/{dataset}/feats.npy")
#     if load_topo:
#         feats_t = loadndarray(f"{os.environ['PROJECT_HOME']}/data/{dataset}/feats_t.npy")
#     labels = loadjson(f"{os.environ['PROJECT_HOME']}/data/{dataset}/class_map.json")
#
#     feats = th.FloatTensor(feats)
#     if load_topo:
#         feats_t = th.FloatTensor(feats_t)
#     labels = process_class_map(G, labels)
#     labels = th.FloatTensor(labels)
#
#     num_features = feats.shape[1]
#     if load_topo:
#         num_features_t = feats_t.shape[1]
#     num_labels = labels.shape[1]
#
#     g = DGLGraph()
#     g.from_networkx(G)
#
#     # print some statistics
#     labels_index = convert_multitarget2_class_index(labels)
#     if load_topo:
#         return g, feats, feats_t, labels, labels_index, num_features, num_features_t, num_labels
#     else:
#         return g, feats, labels, labels_index, num_features, num_labels


if __name__ == '__main__':
    load_data('as_small')
