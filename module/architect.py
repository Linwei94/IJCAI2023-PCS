import logging

import numpy as np
import torch

from torch.nn import functional as F

from module.estimator.utils import arch_matrix_to_graph


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, momentum, weight_decay,
                 arch_learning_rate, arch_weight_decay,
                 predictor, pred_learning_rate,
                 architecture_criterion=F.mse_loss,
                 predictor_criterion=F.mse_loss,
                 reconstruct_criterion=None,
                 preprocessor=None,
                 arch_optim="adam",
                 args=None):
        self.network_momentum = momentum
        self.network_weight_decay = weight_decay
        self.args = args

        # models
        self.model = model
        self.predictor = predictor
        self.preprocessor = preprocessor
        self.reconstruct_criterion = reconstruct_criterion

        if arch_optim == "SGD":
            # architecture optimization
            self.architecture_optimizer = torch.optim.SGD(
                self.model.arch_parameters(), lr=arch_learning_rate,
                weight_decay=arch_weight_decay
            )
        elif arch_optim == "adam":
            # architecture optimization
            self.architecture_optimizer = torch.optim.Adam(
                self.model.arch_parameters(), lr=arch_learning_rate, betas=(0.5, 0.999),
                weight_decay=arch_weight_decay
            )

        self.architecture_criterion = architecture_criterion

        # predictor optimization
        self.predictor_optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=pred_learning_rate, betas=(0.5, 0.999)
        )

        self.predictor_criterion = predictor_criterion


    def predictor_step(self, x, y):
        # clear prev gradient
        self.predictor_optimizer.zero_grad()
        # get output
        y_pred = self.predictor(x)
        # calculate loss
        loss = self.predictor_criterion(y_pred, y)
        # back-prop and optimization step
        loss.backward()
        self.predictor_optimizer.step()
        return y_pred, loss

    def step(self):
        self.architecture_optimizer.zero_grad()
        loss, y_pred = self._backward_step()
        loss.backward()
        self.architecture_optimizer.step()
        return loss, y_pred

    def _backward_step(self):
        y_pred = self.predictor(self.model.arch_weights(cat=False).unsqueeze(0))
        if self.args.acceceloss:
            target = torch.ones_like(y_pred)
            target[:,1] = 0
        else:
            target = torch.zeros_like(y_pred)
        loss = self.architecture_criterion(y_pred, target)
        return loss, y_pred

