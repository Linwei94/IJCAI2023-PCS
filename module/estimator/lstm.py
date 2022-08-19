import torch
from torch import nn


class LSTMExtractor(nn.Module):

    def __init__(self, input_dim, latent_dim, num_layers=1):
        super(LSTMExtractor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=latent_dim,
            batch_first=True,
            num_layers=num_layers
        )

    def forward(self, opt, adj):
        out, (hidden, cell) = self.lstm(torch.cat([opt, adj], dim=-1))
        return hidden.squeeze()
