'''
Pytorch implementation of ResNet models.

Reference:
[1] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR, 2016.
'''
import torch.nn.functional as F
from typing import Union, List
import copy

import torch
from torch import nn, autograd
from utils import gumbel_like
from utils import gumbel_softmax_v1 as gumbel_softmax
from .resnet_origin import resnet50 as resnet50_origin
import data.cifar10 as cifar10
import torch.backends.cudnn as cudnn





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):

    def __init__(self, block, num_blocks, criterion=None, tau=0.1, num_classes=10, temp=1.0, weight_root=None):
        """
        :param C: init channels number
        :param num_classes: classes numbers
        :param layers: total number of layers
        :param criterion: loss function
        :param steps:
        :param multiplier:
        :param stem_multiplier:
        """
        super(ResNet, self).__init__()
        self._block = block
        self._num_blocks = num_blocks
        self._num_classes = num_classes
        self._criterion = criterion
        self._tau = tau
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.temp = temp
        self.block_names = ['layer1','layer2','layer3','layer4','fc']
        self.weight_root = weight_root
        if num_blocks == [2, 2, 2, 2] and block == type(BasicBlock(1,1)):
            self.model_name = "resnet18"
        elif num_blocks == [3, 4, 6, 3] and block == type(BasicBlock(1,1)):
            self.model_name = "resnet34"
        elif num_blocks == [3, 4, 6, 3] and block == type(Bottleneck(1,1)):
            self.model_name = "resnet50"
        elif num_blocks == [3, 4, 23, 3] and block == type(Bottleneck(1,1)):
            self.model_name = "resnet101"
        elif num_blocks == [3, 4, 26, 3] and block == type(Bottleneck(1,1)):
            self.model_name = "resnet110"
        elif num_blocks == [3, 8, 36, 3] and block == type(Bottleneck(1,1)):
            self.model_name = "resnet152"

        self.ops = None




    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def new(self):
        model_new = ResNet(self._block, self._num_blocks, self._layers, self._criterion).to('cuda')
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def load_gumbel_weight(self):
        weights = self.arch_weights(cat=False)
        index = torch.argmax(weights, -1)

        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name, 350)
        model_dict = torch.load(str(saved_model_name))
        self.load_state_dict(model_dict, strict=False)

        for block_name in self.block_names:
            idx = self.block_names.index(block_name)
            saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name, self.ops[index[idx]])
            model_dict = torch.load(str(saved_model_name))
            if block_name == 'fc':
                modified_dict = {k[3:]: v for k, v in model_dict.items() if k.startswith(block_name)}
                model_dict.update(modified_dict)
            else:
                modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith(block_name)}
                model_dict.update(modified_dict)
            getattr(self, block_name).load_state_dict(model_dict, strict=False)


        return self.ops[index.cpu()]

    def load_combination(self, index):

        saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name, 350)
        model_dict = torch.load(str(saved_model_name))
        self.load_state_dict(model_dict, strict=False)

        for block_name in self.block_names:
            idx = self.block_names.index(block_name)
            saved_model_name = "{}/{}_cross_entropy_{}.model".format(self.weight_root, self.model_name, self.ops[index[idx]])
            model_dict = torch.load(str(saved_model_name))
            if block_name == 'fc':
                modified_dict = {k[3:]: v for k, v in model_dict.items() if k.startswith(block_name)}
                model_dict.update(modified_dict)
            else:
                modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith(block_name)}
                model_dict.update(modified_dict)
            getattr(self, block_name).load_state_dict(model_dict, strict=False)


        return self.ops[index.cpu()]

    def load_combination_weight(self, combination, weight_folder, model_name):
        if -1 not in combination:
            saved_model_name = "{}/{}_cross_entropy_{}.model".format(weight_folder, model_name, 350)
            model_dict = torch.load(str(saved_model_name))
            # modified_dict = {k[7:]: v for k, v in model_dict.items()}
            # model_dict.update(modified_dict)
            self.load_state_dict(model_dict, strict=True)

        for block_name in self.block_names:
            idx = self.block_names.index(block_name)
            if not combination[idx] == -1:
                saved_model_name = "{}/{}_cross_entropy_{}.model".format(weight_folder, model_name,
                    combination[idx])
                model_dict = torch.load(str(saved_model_name))
                if block_name == 'fc':
                    modified_dict = {k[3:]: v for k, v in model_dict.items() if k.startswith(block_name)}
                    model_dict.update(modified_dict)
                else:
                    modified_dict = {k[7:]: v for k, v in model_dict.items() if k.startswith(block_name)}
                    model_dict.update(modified_dict)
                getattr(self, block_name).load_state_dict(model_dict, strict=False)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out) / self.temp
        return out

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def initialize_alphas(self):
        # number of layers
        k = 5
        # number of candidates
        num_ops = len(self.ops)

        # init architecture parameters alpha
        self.alphas = (1e-3 * torch.randn(k, num_ops)).to('cuda').requires_grad_(True)
        # init Gumbel distribution for Gumbel softmax sampler
        self.gumbel = gumbel_like(self.alphas)


    def arch_parameters(self) -> List[torch.tensor]:
        return [self.alphas]

    def arch_weights(self, smooth=False, cat: bool=True) -> Union[List[torch.tensor], torch.tensor]:
        if smooth:
            # TODO
            pass
        weights = gumbel_softmax(self.alphas, tau=self._tau, dim=-1, g=self.gumbel)
        if cat:
            return torch.cat(weights)
        else:
            return weights


    def smooth_alpha(self):
        smooth_factor = 0.9
        alpha_base = copy.deepcopy(self.alphas)
        alpha_temp = copy.deepcopy(self.alphas)

        for i in range(1,10):
            temp = torch.zeros_like(alpha_base)
            temp[:,i:] = alpha_base[:,:-i] * (smooth_factor / i)
            alpha_temp = alpha_temp + temp
        for i in range(1,10):
            temp = torch.zeros_like(alpha_base)
            temp[:,:-i] = alpha_base[:,i:] * (smooth_factor / i)
            alpha_temp = alpha_temp + temp

        self.alphas.data = alpha_temp



def resnet18(temp=1.0, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], temp=temp, **kwargs)
    return model


def resnet34(temp=1.0, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], temp=temp, **kwargs)
    return model


def resnet50(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], temp=temp, **kwargs)
    return model


def resnet101(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], temp=temp, **kwargs)
    return model


def resnet110(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 26, 3], temp=temp, **kwargs)
    return model


def resnet152(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], temp=temp, **kwargs)
    return model