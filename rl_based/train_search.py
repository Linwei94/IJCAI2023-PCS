import logging
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision.datasets import CIFAR10

import utils
from genotypes import Genotype, PRIMITIVES
from module.estimator.memory import Memory
from module.estimator.predictor import Predictor, weighted_loss
from rl_based.controller import Controller
from rl_based.model_search_rl import Network
from utils import AverageMeter, gpu_usage


CIFAR_CLASSES = 10


def main():

    args = utils.get_search_arguments()

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # enable GPU and set random seeds
    np.random.seed(args.seed)                  # set random seed: numpy
    torch.cuda.set_device(args.gpu)

    # NOTE: "deterministic" and "benchmark" are set for reproducibility
    # such settings have impacts on efficiency
    # for speed test, disable "deterministic" and enable "benchmark"
    # reproducible search
    # cudnn.deterministic = True
    # cudnn.benchmark = False
    # fast search
    cudnn.deterministic = False
    cudnn.benchmark = True

    torch.manual_seed(args.seed)               # set random seed: torch
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)          # set random seed: torch.cuda
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # use cross entropy as loss function
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to('cuda')

    # CNN model
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.to('cuda')
    logging.info("model param size = %fMB", utils.count_parameters_in_MB(model))

    # use SGD to optimize the model (optimize model.parameters())
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    # rl model
    rl_model = Controller(num_ops=8-1, num_cmp=4, lstm_hidden=64) # num_ops - 1: do not use `none`
    rl_model = rl_model.to(rl_model.device)
    rl_optimizer = torch.optim.Adam(rl_model.parameters(), lr=0.0035, betas=(0.1, 0.999), eps=1e-3)

    # construct data transformer (including normalization, augmentation)
    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    # load cifar100 data training set (train=True)
    train_data = CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    # generate data indices
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    # split training set and validation queue given indices
    # train queue:
    train_queue = DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.num_workers)

    # validation queue:
    valid_queue = DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.num_workers)

    # learning rate scheduler (with cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(args.epochs), eta_min=args.learning_rate_min)

    memory = Memory(args.memory_size, args.predictor_batch_size)

    # --- Part 1: model warm-up and build memory---
    # 1.1 model warm-up
    if args.load_model is not None:
        # load from file
        logging.info('Load warm-up from %s', args.load_model)
        model.load_state_dict(torch.load(os.path.join(args.load_model, 'model-weights-warm-up.pt')))
        warm_up_seqs = utils.pickle_load(os.path.join(args.load_model, 'seqs-warm-up.pickle'))
    else:
        # 1.1.1 sample cells for warm-up
        warm_up_seqs = []
        # assert args.warm_up_population >= args.predictor_batch_size
        rl_model.train()
        for epoch in range(args.warm_up_population):
            arch_seqs, _, _ = rl_model()
            warm_up_seqs.append(arch_seqs)
        utils.pickle_save(warm_up_seqs, os.path.join(args.save, 'seqs-warm-up.pickle'))
        # 1.1.2 warm up
        for epoch, arch_seqs in enumerate(warm_up_seqs):
            logging.info('[warm-up model] epoch %d/%d', epoch + 1, args.warm_up_population)
            # warm-up
            objs, top1 = model_train(queue=train_queue, model=model, arch_seqs=arch_seqs,
                                     criterion=criterion, optimizer=optimizer)
            logging.info('[warm-up model] epoch %d/%d overall loss=%.4f top1-acc=%.4f' %
                         (epoch + 1, args.warm_up_population, objs, top1))
            # save weights
            utils.save(model, os.path.join(args.save, 'model-weights-warm-up.pt'))
            # gpu info
            gpu_usage()

    # 1.2 build memory (i.e. valid model)
    if args.load_memory is not None:
        logging.info('Load valid model from %s', args.load_model)
        model.load_state_dict(torch.load(os.path.join(args.load_memory, 'model-weights-valid.pt')))
        memory.memory = utils.pickle_load(os.path.join(args.load_memory, 'memory-warm-up.pickle'))
    else:
        for epoch, arch_seqs in enumerate(warm_up_seqs):
            # train model for one step
            objs, top1 = model_train(queue=train_queue, model=model, arch_seqs=arch_seqs,
                                     criterion=criterion, optimizer=optimizer)
            logging.info('[build memory] train model-%03d loss=%.4f top1-acc=%.4f',
                         epoch + 1, objs, top1)
            # valid model
            objs, top1 = model_valid(queue=valid_queue, model=model, arch_seqs=arch_seqs, criterion=criterion)
            logging.info('[build memory] valid model-%03d loss=%.4f top1-acc=%.4f',
                         epoch + 1, objs, top1)
            # save to memory
            memory.append(weights=model.arch_weights().detach(),
                          loss=torch.tensor(objs, dtype=torch.float32).to('cuda'))
            # checkpoint: model, memory
            utils.save(model, os.path.join(args.save, 'model-weights-valid.pt'))
            utils.pickle_save(memory.memory, os.path.join(args.save, 'memory-warm-up.pickle'))

    logging.info('memory size=%d', len(memory))

    # --- Part 2 predictor warm-up ---
    # init predictor
    predictor = Predictor(input_size=4 * 2, hidden_size=args.predictor_hidden_state).to('cuda')
    # predictor loss
    if not args.weighted_loss:
        logging.info('using MSE loss')
        predictor_criterion = nn.MSELoss().to('cuda')
    else:
        logging.info('using weighted MSE loss')
        predictor_criterion = weighted_loss
    # predictor optimizer
    predictor_optimizer = torch.optim.Adam(predictor.parameters(),
                                           lr=args.pred_learning_rate,
                                           betas=(0.5, 0.999))
    for epoch in range(args.predictor_warm_up):
        # warm-up
        p_loss, p_true, p_pred = predictor_train(predictor, predictor_optimizer, predictor_criterion, memory)
        if epoch % args.report_freq == 0 or epoch == args.predictor_warm_up:
            logging.info('[warm-up predictor] epoch %d/%d loss=%.4f', epoch, args.predictor_warm_up, p_loss)
            logging.info('\np-true: %s\np-pred: %s', p_true.data, p_pred.data)
            # save predictor
            utils.save(predictor, os.path.join(args.save, 'predictor-warm-up.pt'))
    # gpu info
    gpu_usage()

    # --- Part 3 architecture search ---
    baseline = 0.0
    for epoch in range(args.epochs):
        # get current learning rate
        lr = scheduler.get_lr()[0]
        logging.info('[architecture search] epoch %d/%d lr %e' % (epoch + 1, args.epochs, lr))
        # search
        baseline = architecture_search(epoch=epoch, train_queue=train_queue, valid_queue=valid_queue,
                                       model=model, criterion=criterion, optimizer=optimizer,
                                       rl_model=rl_model, rl_optimizer=rl_optimizer, baseline=baseline,
                                       predictor=predictor, predictor_optimizer=predictor_optimizer,
                                       predictor_criterion=predictor_criterion, memory=memory,
                                       predictor_epoch=args.predictor_warm_up)
        # save weights
        utils.save(model, os.path.join(args.save, 'model-weights-search.pt'))
        # update learning rate
        scheduler.step()
        # gpu info
        gpu_usage()


def parse_geno(arch_seqs, multiplier=4):
    arch = []
    for seq in arch_seqs:
        cell = []
        for idx, opt in zip(*seq):
            cell.append((PRIMITIVES[opt + 1], idx))
        arch.append(cell)
    return Genotype(normal=arch[0], normal_concat=range(2, 2 + multiplier),
                    reduce=arch[1], reduce_concat=range(2, 2 + multiplier))


def model_train(queue, model, arch_seqs, criterion, optimizer, report_freq=50):
    model.train()
    # create metrics
    objs = AverageMeter()
    top1 = AverageMeter()
    # training loop
    total_steps = len(queue)
    for step, (x, target) in enumerate(queue):
        n = x.size(0)
        # data to CUDA
        x = x.to('cuda').requires_grad_(False)
        target = target.to('cuda', non_blocking=True).requires_grad_(False)
        # clear gradient
        optimizer.zero_grad()
        # predict
        logits = model(input=x, arch_seqs=arch_seqs)
        # calculate loss
        loss = criterion(logits, target)
        # backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
        optimizer.step()
        # calculate accuracy
        acc, = utils.accuracy(logits, target, topk=(1,))
        # update metrics
        objs.update(loss.data.item(), n)
        top1.update(acc.data.item(), n)
        # log progress
        if step % report_freq == 0:
            logging.info('train model %03d/%03d loss=%.4f top1-acc=%.4f' % (step, total_steps, objs.avg, top1.avg))
    # return average metrics
    return objs.avg, top1.avg

def predictor_train(predictor, predictor_optimizer, predictor_criterion, memory):
    predictor.train()
    # create metrics
    objs = AverageMeter()
    batch = memory.get_batch()
    # training loop
    all_y = []
    all_p = []
    for x, y in batch:
        n = x.size(0)

        predictor_optimizer.zero_grad()

        pred = predictor(x)
        loss = predictor_criterion(pred, y)
        loss.backward()

        predictor_optimizer.step()

        objs.update(loss.data.item(), n)
        all_y.append(y)
        all_p.append(pred)

    return objs.avg, torch.cat(all_y), torch.cat(all_p)

def model_valid(queue, model, arch_seqs, criterion, report_freq=50):
    model.eval()
    # create metrics
    objs = AverageMeter()
    top1 = AverageMeter()
    # training loop
    total_steps = len(queue)
    for step, (x, target) in enumerate(queue):
        n = x.size(0)
        # data to CUDA
        x = x.to('cuda').requires_grad_(False)
        target = target.to('cuda', non_blocking=True).requires_grad_(False)
        # predict
        logits = model(input=x, arch_seqs=arch_seqs)
        # calculate loss
        loss = criterion(logits, target)
        # calculate accuracy
        acc, = utils.accuracy(logits, target, topk=(1,))
        # update metrics
        objs.update(loss.data.item(), n)
        top1.update(acc.data.item(), n)
        # log progress
        if step % report_freq == 0:
            logging.info('valid model %03d/%03d loss=%.4f top1-acc=%.4f' % (step, total_steps, objs.avg, top1.avg))
    # return average metrics
    return objs.avg, top1.avg


def architecture_search(epoch, train_queue, valid_queue, model, criterion, optimizer, rl_model, rl_optimizer, baseline,
                        predictor, predictor_optimizer, predictor_criterion, memory, predictor_epoch, bl_dec=1e-3):
    # set the rl_model to train mode
    rl_model.train()
    # clear gradient of the rl_optimizer
    rl_optimizer.zero_grad()

    # -- sample an architecture --
    arch_seqs, entropy, log_prob = rl_model()
    print(arch_seqs)
    logging.info('Genotype: %s' % str(parse_geno(arch_seqs)))

    # -- train the architecture --
    objs, top1 = model_train(queue=train_queue, model=model, arch_seqs=arch_seqs,
                             criterion=criterion, optimizer=optimizer)
    logging.info('[overall] epoch %03d train model loss=%.4f top1-acc=%.4f' % (epoch, objs, top1))

    # -- valid the architecture --
    with torch.no_grad():
        objs, top1 = model_valid(queue=valid_queue, model=model, arch_seqs=arch_seqs, criterion=criterion)
    logging.info('[overall] epoch %03d valid model loss=%.4f top1-acc=%.4f' % (epoch, objs, top1))
    reward = top1

    # -- predictor train --
    p_loss, p_true, p_pred = None, None, None
    for epoch in range(predictor_epoch):
        # warm-up
        p_loss, p_true, p_pred = predictor_train(predictor, predictor_optimizer, predictor_criterion, memory)
        if p_loss < 1e-3: break
    logging.info('[architecture search] train predictor p_loss=%.4f\np-true: %s\np-pred: %s'
                 % (p_loss, p_true.data, p_pred.data))

    # -- controller update --
    # calculate REINFORCEMENT loss
    baseline -= bl_dec * (baseline - reward)
    loss = log_prob.sum() * (reward - baseline)
    # back prop through the rl_model
    loss.backward()
    # apply optimization
    rl_optimizer.step()

    # log
    print('resnet50_archlamda25_gs1e-1_s1=%.4f, loss=%.4f, reward=%.4f' % (baseline, loss.item(), reward))
    logging.info('[rl_model] epoch %03d loss=%.4f' % (epoch, loss.item()))

    return baseline


if __name__ == '__main__':
    main()
