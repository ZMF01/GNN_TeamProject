from collections import OrderedDict

import torch as th
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
import numpy as np
import torch.nn.functional as F


class GraphSageConv(nn.Module):
    def __init__(self, *, in_feats, out_feats, aggregator_type, feat_drop,
                 activation=None):
        """

        :param kwargs: projection used to denote input and output projection layer, whose output_dim != input_dim
        """
        super(GraphSageConv, self).__init__()
        aggregator_types = {'mean'}
        assert aggregator_type in aggregator_types
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.feats_dropout = nn.Dropout(feat_drop)
        self.activation = activation
        self.W_l = nn.Linear(2*self.in_feats, self.out_feats, bias=True)

        self.initialize_parameters()

    def initialize_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W_l.weight, gain=gain)

    def forward(self, G, feats):
        G = G.local_var()
        h = self.feats_dropout(feats)

        G.ndata["h"] = h
        G.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neighbors'))  # gc label
        h_neigh = G.ndata['neighbors']

        # t = th.cat((h, self.feats_t), dim=1
        # h_1 = self.W_3(h_1)

        h_combine = th.cat((h, h_neigh), dim=1)
        h = self.W_l(h_combine)
        # h = self.gru(h_combine, feats)

        return h