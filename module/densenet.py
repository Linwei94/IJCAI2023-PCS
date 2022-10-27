'''
Pytorch impplementation of DenseNet.

Reference:
[1] Gao Huang, Zhuang Liu, and Kilian Q. Weinberger. Densely connected convolutional networks.
    arXiv preprint arXiv:1608.06993, 2016a.
'''

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List
from utils import gumbel_like
from utils import gumbel_softmax_v1 as gumbel_softmax


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, temp=1.0, criterion=None, tau=0.1, weight_root=None):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.temp = temp

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

        self._tau = tau
        self.weight_root = weight_root
        self.model_name = "densenet121"
        self._initialize_alphas()


    def load_combination_weight(self, combination, weight_folder, model_name):
        self.weight_root = weight_folder
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name, 350)
        model_dict = torch.load(str(saved_model_name))
        self.load_state_dict(model_dict, strict=True)

        # layer 1
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name,
                                                                 combination[0] + 1)
        model_dict = torch.load(str(saved_model_name))
        modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith('dense1')}
        modified_dict2 = {k[7:]: v for k, v in model_dict.items() if k.startswith('trans1')}
        model_dict.update(modified_dict)
        model_dict.update(modified_dict2)
        getattr(self, 'dense1').load_state_dict(model_dict, strict=False)
        getattr(self, 'trans1').load_state_dict(model_dict, strict=False)

        # layer 2
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name,
                                                                 combination[1] + 1)
        model_dict = torch.load(str(saved_model_name))
        modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith('dense2')}
        modified_dict2 = {k[7:]: v for k, v in model_dict.items() if k.startswith('trans2')}
        model_dict.update(modified_dict)
        model_dict.update(modified_dict2)
        getattr(self, 'dense2').load_state_dict(model_dict, strict=False)
        getattr(self, 'trans2').load_state_dict(model_dict, strict=False)

        # layer 3
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name,
                                                                 combination[2] + 1)
        model_dict = torch.load(str(saved_model_name))
        modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith('dense3')}
        modified_dict2 = {k[7:]: v for k, v in model_dict.items() if k.startswith('trans3')}
        model_dict.update(modified_dict)
        model_dict.update(modified_dict2)
        getattr(self, 'dense3').load_state_dict(model_dict, strict=False)
        getattr(self, 'trans3').load_state_dict(model_dict, strict=False)

        # layer 4
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name,
                                                                 combination[3] + 1)
        model_dict = torch.load(str(saved_model_name))
        modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith('dense4')}
        modified_dict2 = {k[3:]: v for k, v in model_dict.items() if k.startswith('bn')}
        model_dict.update(modified_dict)
        model_dict.update(modified_dict2)
        getattr(self, 'dense4').load_state_dict(model_dict, strict=False)
        getattr(self, 'bn').load_state_dict(model_dict, strict=False)

        # layer 4
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name,
                                                                 combination[4] + 1)
        model_dict = torch.load(str(saved_model_name))
        modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith('linear')}
        model_dict.update(modified_dict)
        getattr(self, 'linear').load_state_dict(model_dict, strict=False)

    def load_combination(self, combination):
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name, 350)
        model_dict = torch.load(str(saved_model_name))
        self.load_state_dict(model_dict, strict=True)

        # layer 1
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name,
                                                                 combination[0] + 1)
        model_dict = torch.load(str(saved_model_name))
        modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith('dense1')}
        modified_dict2 = {k[7:]: v for k, v in model_dict.items() if k.startswith('trans1')}
        model_dict.update(modified_dict)
        model_dict.update(modified_dict2)
        getattr(self, 'dense1').load_state_dict(model_dict, strict=False)
        getattr(self, 'trans1').load_state_dict(model_dict, strict=False)

        # layer 2
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name,
                                                                 combination[1] + 1)
        model_dict = torch.load(str(saved_model_name))
        modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith('dense2')}
        modified_dict2 = {k[7:]: v for k, v in model_dict.items() if k.startswith('trans2')}
        model_dict.update(modified_dict)
        model_dict.update(modified_dict2)
        getattr(self, 'dense2').load_state_dict(model_dict, strict=False)
        getattr(self, 'trans2').load_state_dict(model_dict, strict=False)

        # layer 3
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name,
                                                                 combination[2] + 1)
        model_dict = torch.load(str(saved_model_name))
        modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith('dense3')}
        modified_dict2 = {k[7:]: v for k, v in model_dict.items() if k.startswith('trans3')}
        model_dict.update(modified_dict)
        model_dict.update(modified_dict2)
        getattr(self, 'dense3').load_state_dict(model_dict, strict=False)
        getattr(self, 'trans3').load_state_dict(model_dict, strict=False)

        # layer 4
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name,
                                                                 combination[3] + 1)
        model_dict = torch.load(str(saved_model_name))
        modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith('dense4')}
        modified_dict2 = {k[3:]: v for k, v in model_dict.items() if k.startswith('bn')}
        model_dict.update(modified_dict)
        model_dict.update(modified_dict2)
        getattr(self, 'dense4').load_state_dict(model_dict, strict=False)
        getattr(self, 'bn').load_state_dict(model_dict, strict=False)

        # layer 4
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name,
                                                                 combination[4] + 1)
        model_dict = torch.load(str(saved_model_name))
        modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith('linear')}
        model_dict.update(modified_dict)
        getattr(self, 'linear').load_state_dict(model_dict, strict=False)
    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out) / self.temp
        return out

    def _initialize_alphas(self):
        # number of layers
        k = 5
        # number of candidates
        num_ops = 350

        # init architecture parameters alpha
        self.alphas = (1e-3 * torch.randn(k, num_ops)).to('cuda').requires_grad_(True)
        # init Gumbel distribution for Gumbel softmax sampler
        self.gumbel = gumbel_like(self.alphas)

    def arch_weights(self, smooth=False, cat: bool=True) -> Union[List[torch.tensor], torch.tensor]:
        weights = gumbel_softmax(self.alphas, tau=self._tau, dim=-1, g=self.gumbel)
        if cat:
            return torch.cat(weights)
        else:
            return weights


    def load_gumbel_weight(self):
        weights = self.arch_weights(cat=False)
        combination = torch.argmax(weights, -1)
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name, 350)
        model_dict = torch.load(str(saved_model_name))
        self.load_state_dict(model_dict, strict=True)

        # layer 1
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name,
                                                                 combination[0] + 1)
        model_dict = torch.load(str(saved_model_name))
        modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith('dense1')}
        modified_dict2 = {k[7:]: v for k, v in model_dict.items() if k.startswith('trans1')}
        model_dict.update(modified_dict)
        model_dict.update(modified_dict2)
        getattr(self, 'dense1').load_state_dict(model_dict, strict=False)
        getattr(self, 'trans1').load_state_dict(model_dict, strict=False)

        # layer 2
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name,
                                                                 combination[1] + 1)
        model_dict = torch.load(str(saved_model_name))
        modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith('dense2')}
        modified_dict2 = {k[7:]: v for k, v in model_dict.items() if k.startswith('trans2')}
        model_dict.update(modified_dict)
        model_dict.update(modified_dict2)
        getattr(self, 'dense2').load_state_dict(model_dict, strict=False)
        getattr(self, 'trans2').load_state_dict(model_dict, strict=False)

        # layer 3
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name,
                                                                 combination[2] + 1)
        model_dict = torch.load(str(saved_model_name))
        modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith('dense3')}
        modified_dict2 = {k[7:]: v for k, v in model_dict.items() if k.startswith('trans3')}
        model_dict.update(modified_dict)
        model_dict.update(modified_dict2)
        getattr(self, 'dense3').load_state_dict(model_dict, strict=False)
        getattr(self, 'trans3').load_state_dict(model_dict, strict=False)

        # layer 4
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name,
                                                                 combination[3] + 1)
        model_dict = torch.load(str(saved_model_name))
        modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith('dense4')}
        modified_dict2 = {k[3:]: v for k, v in model_dict.items() if k.startswith('bn')}
        model_dict.update(modified_dict)
        model_dict.update(modified_dict2)
        getattr(self, 'dense4').load_state_dict(model_dict, strict=False)
        getattr(self, 'bn').load_state_dict(model_dict, strict=False)

        # layer 4
        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name,
                                                                 combination[4] + 1)
        model_dict = torch.load(str(saved_model_name))
        modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith('linear')}
        model_dict.update(modified_dict)
        getattr(self, 'linear').load_state_dict(model_dict, strict=False)

        return combination

    def arch_parameters(self) -> List[torch.tensor]:
        return [self.alphas]


def densenet121(temp=1.0, **kwargs):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, temp=temp, **kwargs)


def densenet169(temp=1.0, **kwargs):
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, temp=temp, **kwargs)


def densenet201(temp=1.0, **kwargs):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, temp=temp, **kwargs)


def densenet161(temp=1.0, **kwargs):
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48, temp=temp, **kwargs)