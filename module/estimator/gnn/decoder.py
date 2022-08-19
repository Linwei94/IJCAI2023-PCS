import torch
import torch.nn.functional as F
from torch import nn


class LinearDecoder(nn.Module):
    def __init__(self, latent_dim, decode_dim, dropout,
                 activation_adj=torch.sigmoid, activation_opt=torch.sigmoid,
                 adj_hidden_dim=None, opt_hidden_dim=None):
        super(LinearDecoder, self).__init__()
        if adj_hidden_dim is None:
            self.adj_hidden_dim = latent_dim
        if opt_hidden_dim is None:
            self.opt_hidden_dim = latent_dim
        self.activation_adj = activation_adj
        self.activation_opt = activation_opt
        self.linear_opt = nn.Linear(latent_dim, decode_dim)
        self.dropout = dropout

    def forward(self, embedding):
        embedding = F.dropout(embedding, p=self.dropout, training=self.training)
        opt = self.linear_opt(embedding)
        adj = torch.matmul(embedding, embedding.transpose(1, 2))
        return self.activation_opt(opt, -1), self.activation_adj(adj)
