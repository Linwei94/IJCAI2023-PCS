import torch
from torch import nn


def entropy_cal(logits, prob):
    """
    :param logits: network output
    :param prob: softmax of network output
    :return: entropy
    """
    # detach from gradient
    logits = logits.detach()
    prob = prob.detach()
    # calculate log softmax
    log_prob = nn.functional.log_softmax(logits, dim=-1)
    # calculate entropy
    return -torch.mul(log_prob, prob).sum(dim=1).mean()


class Controller(nn.Module):

    def __init__(self, num_ops, num_cmp, lstm_hidden, device='cuda'):
        """
        :param num_ops: number of operation candidates
        :param num_cmp: number of computation nodes in a cell
        :param lstm_hidden: hidden state of lstm
        :param device: where to run the controller, cpu or cuda
        """
        super(Controller, self).__init__()

        self.num_ops = num_ops
        self.num_cmp = num_cmp
        self.lstm_hidden = lstm_hidden
        self.device = device

        # embed node index to vector
        self.embedding = nn.Embedding(num_embeddings=num_ops+1, embedding_dim=lstm_hidden)
        # the lstm cell
        self.lstm = nn.LSTMCell(input_size=lstm_hidden, hidden_size=lstm_hidden)
        # predict which operation candidate
        self.opt_linear = nn.Linear(lstm_hidden, num_ops)
        # predict which index to be connected
        self.w_attn_1 = nn.Linear(in_features=lstm_hidden, out_features=lstm_hidden, bias=False)
        self.w_attn_2 = nn.Linear(in_features=lstm_hidden, out_features=lstm_hidden, bias=False)
        self.v_attn = nn.Linear(in_features=lstm_hidden, out_features=1, bias=False)

    def forward(self, x=None, hc=None):
        normal_cell, normal_entropy, normal_log_prob, prev_hc = self._generate_cell(None, None)
        reduce_cell, reduce_entropy, reduce_log_prob, _ = self._generate_cell(None, prev_hc)
        entropy = normal_entropy + reduce_entropy
        log_prob = normal_log_prob + reduce_log_prob
        return (normal_cell, reduce_cell), entropy, log_prob

    def _generate_cell(self, x=None, hc=None):

        x = x if x is not None else torch.tensor([0]).to(self.device)
        prev_hc = hc

        anchors = []
        anchors_w_1 = []
        idx_seq = []
        opt_seq = []
        entropy = []
        log_prob = []

        # 2 anchors
        x = self.embedding(x)
        for s in range(2):
            h, c = self.lstm(input=x, hx=prev_hc)
            anchors.append(torch.zeros_like(h))
            anchors_w_1.append(self.w_attn_1(h))
            prev_hc = (h, c)

        # generate computation nodes in the cell, each has two ops
        for s in range(2, self.num_cmp + 2):
            # index A/B
            for i in range(2):
                # lstm prop
                h, c = self.lstm(input=x, hx=prev_hc)
                # calculate query for attention
                query = torch.stack(anchors_w_1[:s], dim=1)
                query = query.view(s, self.lstm_hidden)
                # calculate output
                idx_out = torch.tanh(self.w_attn_2(h) + query)
                idx_out = self.v_attn(idx_out)
                idx_out = idx_out.view(1, s)
                # sample according to output
                prob = torch.softmax(idx_out, dim=-1)
                idx_id = torch.multinomial(prob, 1).view(1)
                idx_seq.append(idx_id.item())
                # store h, c
                prev_hc = (h, c)
                # calculate log_prob
                log_prob.append(nn.functional.cross_entropy(idx_out, idx_id))
                # calculate entropy
                entropy.append(entropy_cal(idx_out, prob))
                # get next input
                x = anchors[idx_id].view(1, -1)

            # operation A/B
            for i in range(2):
                h, c = self.lstm(input=x, hx=prev_hc)
                op_out = self.opt_linear(h)
                # sample according to output
                prob = torch.softmax(op_out, dim=-1)
                op_id = torch.multinomial(prob, 1).view(1)
                opt_seq.append(op_id.item())
                # store h, c
                prev_hc = (h, c)
                # calculate log_prob
                log_prob.append(nn.functional.cross_entropy(op_out, op_id))
                # calculate entropy
                entropy.append(entropy_cal(op_out, prob))
                # get next embedding of next input
                x = self.embedding(input=op_id + 1)

            # predict anchor
            h, c = self.lstm(input=x, hx=prev_hc)
            anchors.append(h)
            anchors_w_1.append(self.w_attn_1(h))
            x = self.embedding(torch.tensor([0]).to(self.device))
            # store h, c
            prev_hc = (h, c)

        entropy = sum(entropy)
        log_prob = sum(log_prob)

        return (idx_seq, opt_seq), entropy, log_prob, prev_hc


if __name__ == '__main__':
    def main():
        rl_model = Controller(num_ops=8, num_cmp=4, lstm_hidden=64, device='cpu')
        rl_model = rl_model.to(rl_model.device)
        arch_seqs, entropy, log_prob = rl_model()
        print(arch_seqs, entropy, log_prob)
    main()
