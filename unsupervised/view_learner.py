import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import numpy as np
from unsupervised.learning import GraSTI


class ViewLearner(torch.nn.Module):
    def __init__(self, encoder, net):
        super(ViewLearner, self).__init__()

        self.encoder = encoder
        self.input_dim = self.encoder.out_node_dim
        self.net = net
        self.mlp_edge_model = Sequential(
            Linear(self.input_dim*2, 64),
            ReLU(),
            Linear(64, 1))
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch, x, edge_index, beta, edge_attr, edge_weight, dyn_weight):

        _, node_emb = self.encoder(batch, x, edge_index, beta, edge_weight)
        src, dst = edge_index[0], edge_index[1]
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]

        if dyn_weight is None:
            edge_emb = torch.cat([emb_src, emb_dst], 1)
            edge_logits = self.mlp_edge_model(edge_emb)
            return edge_logits
        else:
            edge_prod = torch.sum(emb_src * emb_dst, dim=1)
            mu, std, edge_logits = self.net.get_mu_std_logits(edge_weight)
            # mu, std, edge_logits = self.net.get_mu_std_logits(edge_weight)
            return edge_logits, mu, std, edge_prod
