import torch
from torch import nn

from module.estimator.memory import Experience  # latency


def weighted_loss(output, target):
    # squared error
    loss = (output - target)**2
    # weighted loss
    loss = torch.ones_like(target) / target * loss
    # calculate mean
    loss = torch.mean(loss)
    return loss


class Predictor(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Predictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(in_features=self.input_size,
                                     out_features=self.hidden_size)

        self.rnn_cell = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                batch_first=True,
                                bidirectional=False,
                                num_layers=1)

        self.logits_cell = nn.Linear(in_features=self.hidden_size,
                                     out_features=1)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, (hidden, cell) = self.rnn_cell(x, hidden)
        out = self.logits_cell(hidden)
        out = torch.sigmoid(out) * 2
        return out.view(-1)

class Predictor2head(nn.Module):

    def __init__(self, input_size, hidden_size, head=2):
        super(Predictor2head, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.head = head

        self.embedding = nn.Linear(in_features=self.input_size,
                                     out_features=self.hidden_size)

        self.rnn_cell = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                batch_first=True,
                                bidirectional=False,
                                num_layers=1)

        self.logits_cell = nn.Linear(in_features=self.hidden_size,
                                     out_features=head)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, (hidden, cell) = self.rnn_cell(x, hidden)
        out = self.logits_cell(hidden)
        out = torch.sigmoid(out) * 2
        return out.view(-1, self.head)
