import logging
import random
from collections import namedtuple, deque

import torch


Experience = namedtuple('Experience', ['weights', 'loss'])


class Memory(object):
    """Memory"""

    def __init__(self, limit=128, batch_size=64, is_gae=False, multiperformance=False):
        assert limit >= batch_size, 'limit (%d) should not less than batch size (%d)' % (limit, batch_size)
        super(Memory, self).__init__()
        self.limit = limit
        self.batch_size = batch_size
        self.memory = deque(maxlen=limit)
        self.is_gae = is_gae
        self.multiperformance = multiperformance

    def get_batch(self, batch_size=None, EA=None):
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size <= self.limit, \
            'require batch_size (%d) exceeds memory limit, should be less than %d' % (batch_size, self.limit)
        length = len(self)
        if batch_size > length:
            logging.warning('required batch_size (%d) is larger than memory size (%d)', batch_size, length)

        indices = [i for i in range(length)]
        random.shuffle(indices)

        weights_normal = []
        weights_reduce = []
        weights = []
        loss = []
        batch = []
        for idx in indices:
            if self.is_gae:
                weights_normal.append(self.memory[idx].weights[0])
                weights_reduce.append(self.memory[idx].weights[1])
            else:
                weights.append(self.memory[idx].weights)
            loss.append(self.memory[idx].loss)
            if len(loss) >= batch_size:
                if self.is_gae:
                    batch.append(((torch.stack(weights_normal), torch.stack(weights_reduce)), torch.stack(loss)))
                elif self.multiperformance:
                    batch.append((torch.stack(weights), torch.stack([acc[0] for acc in loss]),
                                  torch.stack([ece[1] for ece in loss]), torch.stack([item[2] for item in loss])))
                else:
                    batch.append((torch.stack(weights), torch.stack(loss)))
                weights_normal = []
                weights_reduce = []
                weights = []
                loss = []
        if len(loss) > 0:
            if self.is_gae:
                batch.append(((torch.stack(weights_normal), torch.stack(weights_reduce)), torch.stack(loss)))
            elif self.multiperformance:
                batch.append((torch.stack(weights), torch.stack([acc[0] for acc in loss]), torch.stack([ece[1] for ece in loss]), torch.stack([item[2] for item in loss])))
            else:
                batch.append((torch.stack(weights), torch.stack(loss)))
        return batch

    def append(self, weights, loss):
        self.memory.append(Experience(weights=weights, loss=loss))

    def state_dict(self):
        return {'limit': self.limit,
                'batch_size': self.batch_size,
                'memory': self.memory,
                'is_gae': self.is_gae}

    def load_state_dict(self, state_dict):
        self.limit = state_dict['limit']
        self.batch_size = state_dict['batch_size']
        self.memory = state_dict['memory']
        self.is_gae = state_dict['is_gae']

    def __len__(self):
        return len(self.memory)
