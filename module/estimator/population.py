import logging
import random

import torch

from utils import gumbel_softmax_v1 as gumbel_softmax


class Population(object):

    def __init__(self, batch_size, tau, is_gae=False, population_size=100):
        # TODO: is_gae
        super(Population, self).__init__()
        self._individual = []
        self._fitness = []
        self.batch_size = batch_size
        self.tau = tau
        self.population_size = population_size

    def calculate_pareto_front(self):
        self.front = []
        self.domination_count = [0] * len(self._fitness)
        for item in self._fitness:
            for idx, it in enumerate(self._fitness):
                if item[0] > it[0] and item[1] < it[1]:
                    self.domination_count[idx] = self.domination_count[idx] + 1

        front = 0
        while(front <= max(self.domination_count)):
            front_group = []
            for idx, it in enumerate(self.domination_count):
                if it == front:
                    front_group.append(idx)
            if len(front_group) > 0:
                self.front.append(front_group)
            front = front + 1

        return self.front


    def select(self, which):
        if 'highest' == which:
            losses = [item[2] for item in self._fitness]
            index = losses.index(max(losses))
        elif 'lowest' == which:
            losses = [item[2] for item in self._fitness]
            index = losses.index(min(losses))
        elif 'random' == which:
            index = random.randint(0, len(self._fitness)-1)
        elif 'pareto' == which:
            self.calculate_pareto_front()
            indices = self.front[0]
            individuals = [self._individual[idx] for idx in indices]
            fitness = [self._fitness[idx] for idx in indices]
            return indices, individuals, fitness
        elif 'nsga' == which:
            self.calculate_pareto_front()
            indices = []
            for front_i in range(3):
                for item in self.front[front_i]:
                    indices.append(item)
            individuals = [self._individual[idx] for idx in indices]
            fitness = [self._fitness[idx] for idx in indices]
            return indices, individuals, fitness
        else:
            raise ValueError('unknown argument `which`: %s' % which)
        return index, self._individual[index], self._fitness[index]

    def remove(self, which):
        if 'highest' == which:
            losses = [item[2] for item in self._fitness]
            index = losses.index(max(losses))
        elif 'lowest' == which:
            losses = [item[2] for item in self._fitness]
            index = losses.index(min(losses))
        elif 'random' == which:
            index = random.randint(0, len(self._fitness)-1)
        elif 'pareto' == which:
            self.calculate_pareto_front()
            indices = self.front[-1]
            for idx in indices:
                del self._individual[idx]
                del self._fitness[idx]
            return len(indices), indices
        elif 'nsga' == which:
            self.calculate_pareto_front()
            indices = []
            front_i = -1
            while len(self._individual) > self.population_size + len(indices):
                # CROWED SELECT
                # if len(indices) + len(self.front[front_i]) > self.population_size:
                #     break
                for item in self.front[front_i]:
                    indices.append(item)
                front_i = front_i - 1

            for idx in sorted(indices, reverse=True):
                del self._individual[idx]
                del self._fitness[idx]
            return len(indices), indices
        else:
            raise ValueError('unknown argument `which`: %s' % which)
        del self._individual[index]
        del self._fitness[index]
        return index

    def get_batch(self, batch_size=None, tau=None, EA=False):
        if batch_size is None: batch_size = self.batch_size
        if tau is None: tau = self.tau

        length = len(self)
        if batch_size > length:
            logging.warning('required batch_size (%d) is larger than memory size (%d)', batch_size, length)

        indices = [i for i in range(length)]
        random.shuffle(indices)

        if EA:
            individual = []
            acc = []
            ece = []
            valid_loss = []
            batch = []
            for idx in indices:
                (alphas, gumbel) = self._individual[idx]
                weights = gumbel_softmax(alphas, tau=tau, dim=-1, g=gumbel)
                individual.append(weights)
                acc.append(self._fitness[idx][0])
                ece.append(self._fitness[idx][1])
                valid_loss.append(self._fitness[idx][2])
                if len(individual) >= batch_size:
                    batch.append((torch.stack(individual), torch.stack(valid_loss)))
                    individual = []
                    acc = []
                    ece = []
                    valid_loss = []
            if len(individual) > 0:
                batch.append((torch.stack(individual), torch.stack(valid_loss)))
            return batch
        else:
            individual = []
            valid_loss = []
            batch = []
            for idx in indices:
                (alphas, gumbel) = self._individual[idx]
                weights = gumbel_softmax(alphas, tau=tau, dim=-1, g=gumbel)
                individual.append(weights)
                valid_loss.append(self._fitness[idx])
                if len(individual) >= batch_size:
                    batch.append((torch.stack(individual), torch.stack(valid_loss)))
                    individual = []
                    valid_loss = []
            if len(individual) > 0:
                batch.append((torch.stack(individual), torch.stack(valid_loss)))
            return batch

    def append(self, individual, fitness):
        self._individual.append(individual)
        self._fitness.append(fitness)

    def state_dict(self):
        return {'_individual': self._individual,
                '_fitness': self._fitness}

    def load_state_dict(self, state_dict):
        self._individual = state_dict['_individual']
        self._fitness = state_dict['_fitness']

    def __len__(self):
        len_i = len(self._individual)
        len_f = len(self._fitness)
        assert len_i == len_f
        return len_i
