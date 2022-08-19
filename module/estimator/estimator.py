from typing import Union, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F


class PredictorForGraph(nn.Module):

    def __init__(self, in_features, out_features, hidden_features=128):
        super(PredictorForGraph, self).__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        # self.bn_1 = nn.BatchNorm1d(num_features=hidden_features)
        self.linear_2 = nn.Linear(in_features=hidden_features, out_features=out_features)

    def forward(self, z: torch.tensor) -> torch.tensor:
        out = F.relu(z)
        out = self.linear_1(out).transpose(1, 2)
        out = F.relu(out)
        out = F.adaptive_avg_pool1d(out, 1).squeeze()
        out = self.linear_2(out)
        out = torch.sigmoid(out) * 2.0
        return out


class PredictorForLSTM(nn.Module):

    def __init__(self, in_features, out_features):
        super(PredictorForLSTM, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, z: torch.tensor) -> torch.tensor:
        out = self.linear(z)
        # out = torch.relu(out)
        out = torch.sigmoid(out) * 2.0
        return out


class Estimator(nn.Module):

    def __init__(self, extractor: nn.Module,
                 predictor: nn.Module,
                 autoencoding: bool=True):
        super(Estimator, self).__init__()
        self.extractor = extractor
        self.predictor = predictor
        self.autoencoding = autoencoding

    def forward(self, opt: Union[Tuple[torch.tensor], List[torch.tensor]],
                adj: Union[Tuple[torch.tensor], List[torch.tensor]]):
        opt_normal, opt_reduce = opt
        adj_normal, adj_reduce = adj
        if self.autoencoding:
            opt_recon_normal, adj_recon_normal, z_normal = self.extractor(opt_normal, adj_normal)
            opt_recon_reduce, adj_recon_reduce, z_reduce = self.extractor(opt_reduce, adj_reduce)
            z = torch.cat(tensors=(z_normal, z_reduce), dim=-1)
            y = self.predictor(z)
            return (opt_recon_normal, opt_recon_reduce), \
                   (adj_recon_normal, adj_recon_reduce), z, y
        else:
            z_normal = self.extractor(opt_normal, adj_normal)
            z_reduce = self.extractor(opt_reduce, adj_reduce)
            z = torch.cat(tensors=(z_normal, z_reduce), dim=-1)
            y = self.predictor(z)
            return z, y
