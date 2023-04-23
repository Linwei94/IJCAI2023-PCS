import argparse
import glob
import os
import logging
import sys
import time

import numpy as np
import pickle
import torch
import shutil
import torchvision.transforms as transforms

from torch.nn.functional import softmax
import numpy as np
from scipy.stats import laplace, norm
import random


def map_range(li, m_range=(0, 1)):
    li = np.array(li)
    return (li - min(li)) / len(li) * (m_range[1] - m_range[0])

def allSampling(scale=50):
    return np.array(list(range(350)))

def load_txt(path :str) -> list:
    return [line.rstrip('\n') for line in open(path)]

def piecewiseLaplaceSampling(scale=50):
    population_list = [list(range(0, 149)), list(range(149, 249)), list(range(249, 350))]
    size_list = [np.floor(3 / 7 * 50) + 1, np.floor(2 / 7 * 50), np.floor(2 / 7 * 50)]

    samples_list = [0, 1, 2, 3, 4, 149, 150, 151, 152, 153, 249, 250, 251, 252, 253]
    pieces = [0, 149, 249]
    for k, v in enumerate(pieces):
        population = np.array(population_list[k])
        size = np.array(size_list[k])
        population_in_range = map_range(population, (0, 10))
        if k == 0:
            scaled_pdf = laplace.pdf(population_in_range, 0, scale)
        if k == 1:
            scaled_pdf = laplace.pdf(population_in_range, 0, scale) + laplace.pdf(population_in_range, -1490/250, scale)
        if k == 2:
            scaled_pdf = laplace.pdf(population_in_range, 0, scale) + laplace.pdf(population_in_range, -1490/250, scale)\
                         + laplace.pdf(population_in_range, -2490/250, scale)

        normalized_pdf = (scaled_pdf) / sum(scaled_pdf)
        normalized_cmf = np.array([sum(normalized_pdf[:k + 1]) for k, v in enumerate(normalized_pdf)])

        samples = []
        while True:
            _sample = random.random()
            index = 0
            for k, v in enumerate(normalized_cmf):
                if v <= _sample:
                    index = k
                else:
                    break
            if population[index] not in samples_list:
                samples.append(population[index])
            if len(samples) >= size - 5:
                break
        for i in samples:
            samples_list.append(i)

    return np.sort(samples_list)

def laplaceSampling(scale=10):
    population = np.array(list(range(0, 350)))
    size = 50
    population_in_range = map_range(population, (0, 10))
    scaled_pdf = laplace.pdf(population_in_range, 0, scale)

    normalized_pdf = (scaled_pdf) / sum(scaled_pdf)
    normalized_cmf = np.array([sum(normalized_pdf[:k + 1]) for k, v in enumerate(normalized_pdf)])

    samples = []
    while True:
        _sample = random.random()
        index = 0
        for k, v in enumerate(normalized_cmf):
            if v <= _sample:
                index = k
            else:
                break
        if population[index] not in samples:
            samples.append(population[index])
        if len(samples) >= size:
            break

    return np.sort(samples)


def uniformSampling(param):
    return np.sort([int(i * 350/50) for i in range(50)])

def randomSampling(seed=1):
    random.seed(seed)
    return np.sort(random.sample(range(0, 350), 50))

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))

def pickle_save(obj, obj_path):
    with open(obj_path, 'wb') as fp:
        pickle.dump(obj, fp)

def pickle_load(obj_path):
    with open(obj_path, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', script)
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copyfile(script, dst_file)


def gumbel(*size):
    return -(-torch.rand(*size).log()).log()


def gumbel_like(x):
    return -(-torch.rand_like(x).log()).log()

def gumbel_softmax_v2(x, tau=0.1, dim=-1, g=None):
    if g is None:
        g = gumbel_like(x)
    return softmax((g * x + g) / tau, dim=dim)


def gumbel_softmax_v1(x, tau=0.1, dim=-1, g=None):
    if g is None:
        g = gumbel_like(x)
    return softmax((x + g) / tau, dim=dim)


class DimensionImportanceWeight(object):

    def __init__(self, model, v_type='mean'):
        super(DimensionImportanceWeight, self).__init__()
        assert v_type in ('mean', 'sum', 'single'), 'unknown `v_type`' % v_type
        self.model = model
        self.v_type = v_type
        self.num = 0
        self.diw = torch.zeros_like(self.model.alphas)
        self.diw_sum = torch.zeros_like(self.model.alphas)

    def update(self, eps=1e-6):
        self.num += 1
        # square
        grad_sq = self.model.alphas.grad.detach().clone() ** 2
        # norm
        self.diw = grad_sq / (torch.norm(grad_sq, dim=-1).unsqueeze(-1) + eps)
        # sum
        self.diw_sum += self.diw

    def get_diw(self):
        assert self.num > 0, 'diw has not yet been updated'
        if 'mean' == self.v_type:
            return self.diw_sum / self.num
        elif 'sum' == self.v_type:
            return self.diw_sum
        elif 'single' == self.v_type:
            return self.diw
        else:
            raise ValueError('unknown `v_type`' % self.v_type)


def cal_recon_accuracy(opt, adj, opt_recon, adj_recon, threshold=0.5):
    opt_argmax = opt.argmax(dim=-1)
    opt_recon_argmax = opt_recon.argmax(dim=-1)
    opt_acc = (opt_argmax != opt_recon_argmax).sum(dtype=torch.float) / opt_argmax.nelement()

    adj_recon[adj_recon >= threshold] = 1.
    adj_recon[adj_recon < threshold] = 0.
    adj = torch.triu(adj, 1)
    adj_recon = torch.triu(adj_recon, 1)
    b, w, h = adj.size()
    adj_acc = (adj - adj_recon).abs().sum() / (b * (w + 1) * h / 2)
    return (1. - opt_acc) * 100, (1. - adj_acc) * 100


class LRScheduler(object):

    epoch: int = 0

    def __init__(self, optimizer, schedule, total_epochs, lr_min):
        super(LRScheduler, self).__init__()
        self.optimizer = optimizer
        self.schedule = schedule
        self.total_epochs = total_epochs
        self.init_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
        self.lr_min = lr_min

        # first step
        self.step()

    def step(self):
        # update epoch number
        self.epoch += 1

        if self.schedule == 'none':
            return

        # update lr in optimizer
        for idx, param_group in enumerate(self.optimizer.param_groups):

            # get current lr
            lr = param_group['lr']

            # calculate current LR
            if self.schedule == 'trades':
                # schedule as in TRADES paper
                if self.epoch >= 0.75 * self.total_epochs:
                    lr = lr * 0.1
            elif self.schedule == 'trades_fixed':
                if self.epoch >= 0.75 * self.total_epochs:
                    lr = lr * 0.1
                elif self.epoch >= 0.9 * self.total_epochs:
                    lr = lr * 0.01
                elif self.epoch >= self.total_epochs:
                    lr = lr * 0.001
            elif self.schedule == 'cosine':
                # cosine schedule
                lr = self.init_lr[idx] * 0.5 * (1 + np.cos((self.epoch - 1) / self.total_epochs * np.pi))
            elif self.schedule == 'wrn':
                # schedule as in WRN paper
                if self.epoch >= 0.3 * self.total_epochs:
                    lr = lr * 0.2
                elif self.epoch >= 0.6 * self.total_epochs:
                    lr = lr * 0.04
                elif self.epoch >= 0.8 * self.total_epochs:
                    lr = lr * 0.008
            elif self.schedule == 'linear':
                if self.total_epochs - self.epoch > 5:
                    lr = self.init_lr[idx] * (self.total_epochs - 5 - self.epoch) / (self.total_epochs - 6)
                else:
                    # last 5 epochs
                    lr = self.init_lr[idx] * (self.total_epochs - self.epoch) / ((self.total_epochs - 5) * 5)
            elif self.schedule == 'cyclic':
                if self.epoch <= self.total_epochs // 2:
                    lr = self.init_lr[idx] * self.epoch / (self.total_epochs // 2)
                else:
                    lr = self.init_lr[idx] * (self.total_epochs - self.epoch + 1) / (self.total_epochs // 2)
            else:
                # unknown
                raise ValueError('Unknown LR schedule %s' % self.schedule)

            if lr >= self.lr_min or self.schedule in ['cyclic', 'trades', 'trades_fixed']:
                # update lr in optimizer
                param_group['lr'] = lr
            else:
                param_group['lr'] = self.lr_min
                logging.info('reach lr_min')

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def gpu_usage():
    print('Device name: %s' % torch.cuda.get_device_name(0))
    print('Memory usage:')
    print('  Allocated: %.4f GB' % (torch.cuda.memory_allocated(0) / 1024 ** 3))
    print('  Cached:    %.4f GB' % (torch.cuda.memory_reserved(0) / 1024 ** 3))


def get_search_arguments():
    parser = argparse.ArgumentParser("cifar")
    # data
    parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loader workers')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    # save
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    # training setting
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    # search setting
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--memory_size', type=int, default=100, help='size of memory to train predictor')
    parser.add_argument('--warm_up_population', type=int, default=100, help='warm_up_population')
    parser.add_argument('--load_model', type=str, default=None, help='load model weights from file')
    parser.add_argument('--load_memory', type=str, default=None, help='load memory from file')
    parser.add_argument('--predictor_warm_up', type=int, default=500, help='predictor warm-up steps')
    parser.add_argument('--predictor_hidden_state', type=int, default=16, help='predictor hidden state')
    parser.add_argument('--predictor_batch_size', type=int, default=64, help='predictor batch size')
    parser.add_argument('--pred_learning_rate', type=float, default=0.1, help='predictor learning rate')
    parser.add_argument('--tau', type=float, default=0.1, help='tau')
    parser.add_argument('--weighted_loss', action='store_true', default=False, help='use weighted loss')
    # model setting
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    # others
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()

    args.save = 'checkpoints/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    return args


# get weight file name and link with weight index
def get_gdrive_link_with_idx(index):
    name_link = {}
    # read csv file
    with open('google_drive_weight_list.csv', 'r') as f:
        # read one single line
        line = f.readline()
        while line:
            # split line by comma
            line = line.split(',')
            # get weight file name and link
            name_link[line[0]] = line[1]
            # read next line
            line = f.readline()
    # get weight file name
    file_name = f'resnet50_cross_entropy_{index}.model'
    # get weight link
    link = name_link[file_name]
    return file_name, link

get_gdrive_link_with_idx(19)
    