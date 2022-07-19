import time

import dgl
import scipy.io
import urllib.request
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from tqdm import tqdm
from layers import GraphSageConv
import os
from dgl.nn.pytorch import edge_softmax, GATConv


class GraphSage(nn.Module):
    def __init__(self, *, in_dim,
                 num_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSage, self).__init__()

        self.activation = activation

        self.layers = nn.ModuleList()
        # input projection (no residual)
        self.layers.append(GraphSageConv(in_feats=in_dim, out_feats=num_hidden,
                                         aggregator_type=aggregator_type, feat_drop=dropout))
        print(f"Layer {0}: Input Projection: ({in_dim}, {num_hidden})")

        # hidden layers
        for l in range(1, n_layers - 1):
            curr_layer = GraphSageConv(in_feats=num_hidden, out_feats=num_hidden,
                                         aggregator_type=aggregator_type, feat_drop=dropout)
            print(f"Layer {l}: Hidden: ({num_hidden}, {num_hidden})")
            self.layers.append(curr_layer)

        self.layers.append(GraphSageConv(in_feats=num_hidden, out_feats=n_classes,
                                         aggregator_type=aggregator_type, feat_drop=dropout))

    def forward(self, G, feats):
        h = feats

        for i in range(0, len(self.layers) - 1):
            h = self.activation(self.layers[i](G, h))

        # h_combine = th.cat([h, self.feats_t], dim=1)
        # logits = self.layers[-1](h_combine)
        logits = self.layers[-1](G, h)
        return logits
