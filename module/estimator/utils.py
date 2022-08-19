import torch
from torch.nn import functional as F


def stacked_spmm(A, B):
    assert A.dim()==3
    assert B.dim()==3
    return torch.matmul(A, B)


class GraphPreprocessor(object):

    def __init__(self, mode:int=0, lamb:float=0.):
        super(GraphPreprocessor, self).__init__()
        self.mode = mode
        self.lamb = lamb

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def process(self, adj, opt):
        if self.mode == 0:
            # do nothing
            pass
        elif self.mode == 1:
            raise NotImplementedError
        elif self.mode == 2:
            raise NotImplementedError
        elif self.mode == 3:
            # using lambda*A + (1-lambda)*A^T
            adj = self.lamb * adj + (1 - self.lamb) * adj.transpose(-1, -2)
        elif self.mode == 4:
            adj = adj + adj.triu(1).transpose(-1, -2)
        else:
            raise ValueError('Unexpected preprocessing type: %d' % self.mode)

        return adj, opt

    def reverse(self, adj, opt):
        if self.mode == 0:
            pass
        elif self.mode == 1:
            raise NotImplementedError
        elif self.mode == 2:
            raise NotImplementedError
        elif self.mode == 3:
            adj = 1.0 / self.lamb * adj.triu(1)
        elif self.mode == 4:
            adj = adj.triu(1)
        else:
            raise ValueError('Unexpected preprocessing type: %d' % self.mode)
        return adj, opt


def arch_matrix_to_graph(matrix, steps=4):
    """adj matrix format:
              c_k-2, c_k-1, n_0,0, n_0,1, n_1,0, n_1,1, n_1,2, n_2,0, n_2,1, n_2,2, n_2,3, n_3,0, n_3,1, n_3,2, n_3,3, n_3,4,   c_k
     0: c_k-2     0,     0,     x,     0,     x,     0,     0,     x,     0,     0,     0,     x,     0,     0,     0,     0,     0
     1: c_k-1     0,     0,     0,     x,     0,     x,     0,     0,     x,     0,     0,     0,     x,     0,     0,     0,     0
     2: n_0       0,     0,     0,     0,     0,     0,     x,     0,     0,     x,     0,     0,     0,     x,     0,     0,     1
     3: (n_0,1)
     4: n_1       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     x,     0,     0,     0,     x,     0,     1
     5: (n_1,1)
     6: (n_1,2)
     7: n_2       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     x,     1
     8: (n_2,1)
     9: (n_2,2)
    10: (n_2,3)
    11: n_3       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1
    12: (n_3,1)
    13: (n_3,2)
    14: (n_3,3)
    15: (n_3,4)
    16: c_k       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
    """
    assert steps == 4

    k, num_ops = matrix.shape[1:]

    all_adj = []
    all_opt = []

    for m in matrix:
        selected_weights, selected_ops = m.max(-1)

        # adjacency matrix
        adj = torch.zeros(k + 3, k + 3).to(selected_weights.device)

        # steps = 4
        # for i in range(steps):
        #     for j in range(0, (3 + i) * i // 2 + 2):
        #         print(j)

        adj[0, 2] = selected_weights[0]
        adj[1, 3] = selected_weights[1]

        adj[0, 4] = selected_weights[2]
        adj[1, 5] = selected_weights[3]
        adj[2, 6] = selected_weights[4]
        adj[3, 6] = selected_weights[4]

        adj[0, 7] = selected_weights[5]
        adj[1, 8] = selected_weights[6]
        adj[2, 9] = selected_weights[7]
        adj[3, 9] = selected_weights[7]
        adj[4, 10] = selected_weights[8]
        adj[5, 10] = selected_weights[8]
        adj[6, 10] = selected_weights[8]

        adj[0, 11] = selected_weights[9]
        adj[1, 12] = selected_weights[10]
        adj[2, 13] = selected_weights[11]
        adj[3, 13] = selected_weights[11]
        adj[4, 14] = selected_weights[12]
        adj[5, 14] = selected_weights[12]
        adj[6, 14] = selected_weights[12]
        adj[7, 15] = selected_weights[13]
        adj[8, 15] = selected_weights[13]
        adj[9, 15] = selected_weights[13]
        adj[10, 15] = selected_weights[13]

        for i in range(2, k + 2):
            adj[i, k + 2] = 1.0

        # operations
        opt = F.one_hot(selected_ops, num_classes=num_ops)
        opt = F.pad(opt, (2, 1, 2, 1))
        opt[0, 0] = 1
        opt[1, 1] = 1
        opt[-1, -1] = 1
        opt = opt.float()

        all_adj.append(adj)
        all_opt.append(opt)

    return torch.stack(all_adj), torch.stack(all_opt)
