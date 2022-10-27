import argparse
import glob
import logging
import sys
import time
from typing import Optional
import os

from utils import uniformSampling, randomSampling, piecewiseLaplaceSampling, laplaceSampling, allSampling
# Import dataloaders
import data.cifar10 as cifar10
import data.cifar100 as cifar100
import data.tiny_imagenet as tiny_imagenet

# Import network models
from module.resnet import resnet50, resnet110
from module.resnet_tiny_imagenet import resnet50 as resnet50_ti
from module.wide_resnet import wide_resnet_cifar
from module.densenet import densenet121

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
from scipy.stats import kendalltau
from torch.nn import functional as F
from metrics.metrics import expected_calibration_error, test_classification_net

import utils
from module.architect import Architect

from module.estimator.memory import Memory
from module.estimator.predictor import Predictor, Predictor2head, weighted_loss
from utils import gumbel_like, gpu_usage

# Import metrics to compute
from metrics.metrics import test_classification_net_logits
from metrics.metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss, get_logits_labels

# Import temperature scaling and NLL utilities
from temperature_scaling import ModelWithTemperature


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # enable GPU and set random seeds
    np.random.seed(args.seed)  # set random seed: numpy
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = False
    cudnn.deterministic = True

    args.weight_root = "../weights/{}/{}".format(args.dataset_name, args.model_name)
    args.load_memory = "../checkpoints/{}-{}".format(args.dataset_name,
                                                     args.model_name) if not args.load_memory == "build_memory" else None
    # Taking input for the dataset

    CIFAR_CLASSES = 10
    dataset_num_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'tiny_imagenet': 200
    }

    dataset_loader = {
        'cifar10': cifar10,
        'cifar100': cifar100,
        'tiny_imagenet': tiny_imagenet
    }

    # Mapping model name to model function
    models = {
        'resnet50': resnet50,
        'resnet50_ti': resnet50_ti,
        'resnet110': resnet110,
        'wide_resnet': wide_resnet_cifar,
        'densenet121': densenet121,
    }

    # Sampling Strategy
    sampling_strategies = {
        'all': allSampling,
        'uniformSampling': uniformSampling,
        'randomSampling': randomSampling,
        'piecewiseLaplaceSampling': piecewiseLaplaceSampling,
        'laplaceSampling': laplaceSampling
    }

    dataset = args.dataset_name
    num_classes = dataset_num_classes[dataset]
    sampling_strategy = sampling_strategies[args.sampling_strategy]
    torch.manual_seed(args.seed)  # set random seed: torch
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)  # set random seed: torch.cuda
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    if len(unknown_args) > 0:
        logging.warning('unknown_args: %s' % unknown_args)
    else:
        logging.info('unknown_args: %s' % unknown_args)
    # use cross entropy as loss function
    criterion = nn.CrossEntropyLoss(reduction='sum')
    criterion = criterion.to('cuda')

    # build the model with model_search.Network
    logging.info("init arch param")
    model = models[args.model_name](criterion=criterion, tau=args.tau, weight_root=args.weight_root,
                                    num_classes=num_classes)
    model = model.to('cuda')
    model.ops = sampling_strategy(args.sampling_param)
    model.initialize_alphas()
    print("sampling_strategy: {}({})".format(args.sampling_strategy, args.sampling_param), model.ops)
    logging.info("model param size = %fMB", utils.count_parameters_in_MB(model))

    # use SGD to optimize the model (optimize model.parameters())
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.ftlr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False
    )

    args.data_aug = True
    args.test_batch_size = 64

    if (dataset == 'tiny_imagenet'):
        dataset_root = "../datasets/tiny-imagenet-200"
        train_queue = dataset_loader[dataset].get_data_loader(
            root=dataset_root,
            split='train',
            batch_size=args.train_batch_size,
            pin_memory=args.gpu,
            num_workers=1)

        valid_queue = dataset_loader[dataset].get_data_loader(
            root=dataset_root,
            split='val',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu,
            num_workers=1)

        test_queue = dataset_loader[dataset].get_data_loader(
            root=dataset_root,
            split='val',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu,
            num_workers=1)
    else:
        train_queue, valid_queue = dataset_loader[dataset].get_train_valid_loader(
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            random_seed=1,
            pin_memory=args.gpu,
            num_workers=1
        )
        test_queue = dataset_loader[dataset].get_test_loader(
            batch_size=args.test_batch_size,
            pin_memory=args.gpu,
            num_workers=1
        )

    # construct architect with architect.Architect
    _, feature_num = torch.vstack(model.arch_parameters()).shape
    if args.acceceloss:
        is_gae = False
        # -- preprocessor --
        preprocessor = None
        # -- build model --
        predictor = Predictor2head(input_size=feature_num, hidden_size=args.predictor_hidden_state, head=2)
    # -- build model --
    predictor = Predictor(input_size=feature_num, hidden_size=args.predictor_hidden_state)
    predictor = predictor.to('cuda')
    reconstruct_criterion = None
    if args.acceceloss:
        logging.info('using accece loss for predictor')

        def acceceloss(out, target):
            acc_loss = F.mse_loss(out[:, 0], target[:, 0])
            ece_loss = F.mse_loss(out[:, 1], target[:, 1])
            return acc_loss + args.accecelamda * ece_loss

        def arch_acceceloss(out, target):
            acc_loss = F.mse_loss(out[:, 0], target[:, 0])
            ece_loss = F.mse_loss(out[:, 1], target[:, 1])
            return acc_loss + args.arch_accecelamda * ece_loss

        predictor_criterion = acceceloss
        architecture_criterion = arch_acceceloss

    elif args.weighted_loss:
        logging.info('using weighted MSE loss for predictor')
        predictor_criterion = weighted_loss
        architecture_criterion = F.mse_loss
    else:
        logging.info('using MSE loss for predictor')
        predictor_criterion = F.mse_loss
        architecture_criterion = F.mse_loss

    architect = Architect(
        model=model, momentum=args.momentum, weight_decay=args.weight_decay,
        arch_learning_rate=args.arch_learning_rate, arch_weight_decay=args.arch_weight_decay,
        predictor=predictor, pred_learning_rate=args.pred_learning_rate,
        architecture_criterion=architecture_criterion, predictor_criterion=predictor_criterion,
        reconstruct_criterion=reconstruct_criterion,
        arch_optim=args.arch_optim,
        args=args
    )

    memory = Memory(limit=args.memory_size, batch_size=args.predictor_batch_size,
                        multiperformance=True)

    # --- Part 1: model warm-up and build memory---
    # 1.1 model warm-up
    if args.load_model is not None:
        # load from file
        logging.info('Load warm-up from %s', args.load_model)
        model.load_state_dict(torch.load(os.path.join(args.load_model, 'model-weights-warm-up.pt')))
        warm_up_gumbel = utils.pickle_load(os.path.join(args.load_model, 'gumbel-warm-up.pickle'))
    else:
        # 1.1.1 sample cells for warm-up
        warm_up_gumbel = []
        # assert args.warm_up_population >= args.predictor_batch_size
        for epoch in range(args.warm_up_population):
            gumbel = gumbel_like(model.alphas) * args.gumbel_scale
            warm_up_gumbel.append(gumbel)
        utils.pickle_save(warm_up_gumbel, os.path.join(args.save, 'gumbel-warm-up.pickle'))

    # 1.2 build memory (i.e. valid model)
    if args.load_memory is not None:
        logging.info('Load valid model from %s', args.load_model)
        # model.load_state_dict(torch.load(os.path.join(args.load_memory, 'model-weights-valid.pt')))
        memory.load_state_dict(
            utils.pickle_load(
                os.path.join(args.load_memory, 'memory-warm-up.pickle')
            )
        )
    else:
        for epoch, gumbel in enumerate(warm_up_gumbel):
            # re-sample Gumbel distribution
            model.gumbel = gumbel
            index = model.load_gumbel_weight()
            # replace the optimizer weight
            optimizer.param_groups[0]['params'] = [p for p in model.parameters()]
            # train model for one step
            if not args.notrain:
                model_train(train_queue, model, criterion, optimizer, name='build memory')
            # valid model
            # test result
            valid_conf_matrix, valid_acc, valid_labels, valid_predictions, valid_confidences, valid_loss = test_classification_net(
                model,
                valid_queue)

            valid_ece = expected_calibration_error(valid_confidences, valid_predictions, valid_labels, num_bins=15)
            logging.info(
                'EPOCH %d: alpha+gumbel=%s, valid_loss=%.4f, ece=%.4f, acc=%.4f' % (
                    epoch,
                    index.tolist(),
                    valid_loss,
                    valid_ece,
                    valid_acc)
            )

            memory.append(weights=model.arch_weights(cat=False).detach(),
                              loss=(torch.tensor(valid_acc, dtype=torch.float32).to('cuda'),
                                    torch.tensor(valid_ece, dtype=torch.float32).to('cuda'),
                                    torch.tensor(valid_loss, dtype=torch.float32).to('cuda')))
            # checkpoint: model, memory
            utils.save(model, os.path.join(args.save, 'model-weights-valid.pt'))
            utils.pickle_save(memory.state_dict(),
                              os.path.join(args.save, 'memory-warm-up.pickle'))

    logging.info('memory size=%d', len(memory))

    # --- Part 1.2 build arch-performance dict ---
    architect_performance_dict = {}
    for i in memory.memory:
        arch = str(torch.max(i.weights, -1).indices.tolist())
        architect_performance_dict[arch] = i.loss

    # --- Part 2 predictor warm-up ---
    if args.load_extractor is not None:
        logging.info('Load extractor from %s', args.load_extractor)
        architect.predictor.extractor.load_state_dict(torch.load(args.load_extractor)['weights'])

    architect.predictor.train()
    for epoch in range(args.predictor_warm_up):
        epoch += 1
        # warm-up
        if args.acceceloss:
            pred_train_loss, acc_true, ece_true, loss_true, pred_acc, pred_ece = predictor_train(architect, memory)
            if epoch % args.report_freq == 0 or epoch == args.predictor_warm_up:
                logging.info('[warm-up predictor] epoch %d/%d loss=%.4f', epoch, args.predictor_warm_up,
                             pred_train_loss)
                acc_tau = kendalltau(acc_true.detach().to('cpu'), pred_acc.detach().to('cpu'))[0]
                ece_tau = kendalltau(ece_true.detach().to('cpu'), pred_ece.detach().to('cpu'))[0]
                logging.info('acc kendall\'s-tau=%.4f   ece kendall\'s-tau=%.4f', acc_tau, ece_tau)
        else:
            pred_train_loss, acc_true, ece_true, loss_true, pred_loss = predictor_train(architect, memory)
            if epoch % args.report_freq == 0 or epoch == args.predictor_warm_up:
                logging.info('[warm-up predictor] epoch %d/%d loss=%.4f', epoch, args.predictor_warm_up,
                             pred_train_loss)
                tau = kendalltau(loss_true.detach().to('cpu'), pred_loss.detach().to('cpu'))[0]
                logging.info('loss kendall\'s-tau=%.4f', tau)
        # save predictor
        utils.save(architect.predictor, os.path.join(args.save, 'predictor-warm-up.pt'))

    # gpu info
    gpu_usage()

    # --- Part 3 architecture search ---
    for epoch in range(args.epochs):
        # search
        architecture_search(train_queue, valid_queue, test_queue, model, architect,
                            criterion, optimizer, memory, epoch=epoch,
                            architect_performance_dict=architect_performance_dict)
        # save weights
        utils.save(model, os.path.join(args.save, 'model-weights-search.pt'))


def model_train(train_queue, model, criterion, optimizer, name):
    # set model to training model
    model.train()
    # create metrics
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    # training loop
    total_steps = len(train_queue)
    for step, (x, target) in enumerate(train_queue):
        n = x.size(0)
        # data to CUDA
        x = x.to('cuda').requires_grad_(False)
        target = target.to('cuda', non_blocking=True).requires_grad_(False)
        # update model weight
        # forward
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, target)
        # backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        # update metrics
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
    # return average metrics
    return objs.avg, top1.avg, top5.avg


def model_valid(valid_queue, model, criterion, name):
    # set model to evaluation model
    model.eval()
    # create metrics
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    # validation loop
    total_steps = len(valid_queue)
    # ece test
    predictions_list = []
    confidence_vals_list = []
    labels_list = []
    for step, (x, target) in enumerate(valid_queue):
        n = x.size(0)
        # data to CUDA
        x = x.to('cuda').requires_grad_(False)
        target = target.to('cuda', non_blocking=True).requires_grad_(False)
        # valid model
        logits = model(x)
        loss = criterion(logits, target)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        softmax = F.softmax(logits, dim=1)
        confidence_vals, predictions = torch.max(softmax, dim=1)

        # ece
        predictions_list.extend(predictions.cpu().numpy().tolist())
        confidence_vals_list.extend(confidence_vals.detach().cpu().numpy().tolist())
        labels_list.extend(target.cpu().numpy().tolist())

        # update metrics
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
        # log
        # if step % args.report_freq == 0:
        # logging.info('[%s] valid model %03d/%03d loss=%.4f top1-acc=%.4f top5-acc=%.4f',
        #              name, step, total_steps, objs.avg, top1.avg, top5.avg)
    val_ece = expected_calibration_error(confidence_vals_list, predictions_list, labels_list, num_bins=15)

    return objs.avg, top1.avg, top5.avg, val_ece


def predictor_train(architect, memory, unsupervised=False):
    objs = utils.AverageMeter()
    batch = memory.get_batch(EA=args.evolution)
    all_acc = []
    all_ece = []
    all_loss = []
    all_p_acc = []
    all_p_ece = []
    all_p_loss = []
    for x, acc, ece, loss in batch:
        n = acc.size(0)
        if args.acceceloss:
            y_pred, predictor_train_loss = architect.predictor_step(x, torch.swapaxes(torch.stack([acc, ece]), 0, 1),
                                                                    unsupervised=unsupervised, accece=True)
            pred_acc, pred_ece = y_pred[:, 0], y_pred[:, 1]
            objs.update(predictor_train_loss.data.item(), n)
            all_acc.append(acc)
            all_ece.append(ece)
            all_loss.append(loss)
            all_p_acc.append(pred_acc)
            all_p_ece.append(pred_ece)
            return objs.avg, torch.cat(all_acc), torch.cat(all_ece), all_loss, torch.cat(all_p_acc), torch.cat(
                all_p_ece)

        else:
            y_pred, predictor_train_loss = architect.predictor_step(x, loss, unsupervised=unsupervised, accece=True)
            objs.update(predictor_train_loss.data.item(), n)
            all_acc.append(acc)
            all_ece.append(ece)
            all_loss.append(loss)
            all_p_loss.append(y_pred)
            return objs.avg, torch.cat(all_acc), torch.cat(all_ece), torch.cat(all_loss), torch.cat(all_p_loss)


def architecture_search(train_queue, valid_queue, test_queue, model, architect, criterion, optimizer, memory,
                        epoch=None, architect_performance_dict=None):
    # -- train model --
    model.gumbel = gumbel_like(model.alphas) * args.gumbel_scale
    arch_str = str(torch.max(model.arch_weights(cat=False), -1).indices.tolist())

    if arch_str in architect_performance_dict:
        arch_performance = architect_performance_dict[arch_str]
    else:
        model.load_gumbel_weight()
        # train model for one step
        optimizer.param_groups[0]['params'] = [p for p in model.parameters()]
        if not args.notrain:
            model_train(train_queue, model, criterion, optimizer, name='build memory')

        # -- valid model --
        valid_conf_matrix, valid_acc, valid_labels, valid_predictions, valid_confidences, valid_loss = test_classification_net(
            model,
            valid_queue)

        valid_ece = expected_calibration_error(valid_confidences, valid_predictions, valid_labels, num_bins=15)
        arch_performance = (torch.tensor(valid_acc, dtype=torch.float32).to('cuda'),
                            torch.tensor(valid_ece, dtype=torch.float32).to('cuda'),
                            torch.tensor(valid_loss, dtype=torch.float32).to('cuda'))

    # -- test model --
    p_accuracy, p_ece, a_ece, T_opt, test_loss = test_performance(valid_queue, test_queue, model)

    # save validation to memory
    if args.evolution:
        memory.append(individual=(model.alphas.detach().clone(), model.gumbel.detach().clone()),
                      fitness=arch_performance)
        memory.remove('highest')

    else:
        memory.append(weights=model.arch_weights(cat=False).detach(),
                      loss=arch_performance)

    architect_performance_dict[arch_str] = arch_performance
    valid_acc = arch_performance[0]
    valid_ece = arch_performance[1]
    valid_loss = arch_performance[2]

    utils.pickle_save(memory.state_dict(),
                      os.path.join(args.save, 'memory-search.pickle'))

    # -- predictor train --
    architect.predictor.train()
    # use memory to train predictor
    for _ in range(args.predictor_warm_up):
        if args.acceceloss:
            pred_train_loss, acc_true, ece_true, loss_true, pred_acc, pred_ece = predictor_train(architect, memory)
            acc_tau = kendalltau(acc_true.detach().to('cpu'), pred_acc.detach().to('cpu'))[0]
            ece_tau = kendalltau(ece_true.detach().to('cpu'), pred_ece.detach().to('cpu'))[0]
            if acc_tau > 0.95 and ece_tau > 0.95: break
        else:
            pred_train_loss, acc_true, ece_true, loss_true, pred_loss = predictor_train(architect, memory)
            tau = kendalltau(loss_true.detach().to('cpu'), pred_loss.detach().to('cpu'))[0]
            if tau > 0.95: break

    logging.info(
        'EPOCH %d: alpha=%s, alpha+gumbel=%s, valid_acc=%.4f, valid_ece=%.2f, valid_loss=%.4f, test_acc=%.4f, test_ece=%.2f, test_ece(T)=%.2f(%.2f), test_loss=%.4f, alpha_max=%.4f, gumbel_max=%.4f, lr=%.8f' % (
            epoch,
            model.ops[torch.max(architect.model.arch_parameters()[0], -1).indices.tolist()],
            model.ops[torch.max(architect.model.arch_weights(cat=False), -1).indices.tolist()],
            valid_acc,
            valid_ece * 100,
            valid_loss,
            p_accuracy,
            p_ece * 100,
            a_ece * 100,
            T_opt,
            test_loss,
            model.alphas.max(),
            model.gumbel.max(),
            optimizer.state_dict()['param_groups'][0]['lr']
        ))

    # -- architecture update --
    if args.evolution:
        index, weights, fitness = memory.select('lowest')
        alphas, gumbel = weights
        model.gumbel.data = alphas
        model.gumbel = gumbel
    loss, y_pred = architect.step()
    # print("arch_creterion:", loss, "prediction",y_pred)

    if diw is not None:
        diw.update()


def test_performance(val_loader, test_loader, net):
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = ECELoss().cuda()
    adaece_criterion = AdaptiveECELoss().cuda()
    cece_criterion = ClasswiseECELoss().cuda()

    logits, labels = get_logits_labels(test_loader, net)
    conf_matrix, p_accuracy, _, _, _ = test_classification_net_logits(logits, labels)

    p_ece = ece_criterion(logits, labels).item()
    p_adaece = adaece_criterion(logits, labels).item()
    p_cece = cece_criterion(logits, labels).item()
    p_nll = nll_criterion(logits, labels).item()

    scaled_model = ModelWithTemperature(net)
    scaled_model.set_temperature(val_loader, cross_validate='ece')
    T_opt = scaled_model.get_temperature()
    logits, labels = get_logits_labels(test_loader, scaled_model)
    conf_matrix, accuracy, _, _, _ = test_classification_net_logits(logits, labels)

    ece = ece_criterion(logits, labels).item()
    adaece = adaece_criterion(logits, labels).item()
    cece = cece_criterion(logits, labels).item()
    nll = nll_criterion(logits, labels).item()

    return p_accuracy, p_ece, ece, T_opt, p_nll


if __name__ == '__main__':
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
    parser.add_argument('--grad_clip', type=float, default=2, help='gradient clipping')
    parser.add_argument('--train_batch_size', type=int, default=128)
    # memory setting
    parser.add_argument('--num_of_combinations', type=int, default=500, help='num of random ensembled combinations')

    # search setting
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--memory_size', type=int, default=100, help='size of memory to train predictor')
    parser.add_argument('--warm_up_population', type=int, default=100, help='warm_up_population')
    parser.add_argument('--load_model', type=str, default=None, help='load model weights from file')
    parser.add_argument('--load_memory', type=str, default=None, help='load memory from file')
    parser.add_argument('--tau', type=float, default=0.1, help='tau')
    parser.add_argument('--evolution', action='store_true', default=False, help='use weighted loss')
    parser.add_argument('--pareto', action='store_true', default=False, help='use pareto front')
    parser.add_argument('--smooth', action='store_true', default=False, help='use smooth')
    parser.add_argument('--reduce_memory_before_arch_train_by_size', type=int, default=None,
                        help='reduce_memory_before_arch_train')
    parser.add_argument('--reduce_memory_before_arch_train_by_loss_limit', type=float, default=None,
                        help='reduce_memory_before_arch_train_by_loss_limit')
    parser.add_argument('--arch_optim', type=str, default='adam', help='arch_optim')
    parser.add_argument('--sampling_strategy', type=str, default='all')
    parser.add_argument('--sampling_param', type=int, default=10)

    # predictor setting
    parser.add_argument('--predictor_warm_up', type=int, default=500, help='predictor warm-up steps')
    parser.add_argument('--predictor_hidden_state', type=int, default=16, help='predictor hidden state')
    parser.add_argument('--predictor_batch_size', type=int, default=64, help='predictor batch size')
    parser.add_argument('--pred_learning_rate', type=float, default=1e-3, help='predictor learning rate')
    parser.add_argument('--weighted_loss', action='store_true', default=False, help='use weighted loss')
    parser.add_argument('--load_extractor', type=str, default=None, help='load memory from file')
    parser.add_argument('--acceceloss', action='store_true', default=False, help='acceceloss')
    parser.add_argument('--accecelamda', type=float, default=10, help='accecelamda')
    parser.add_argument('--arch_accecelamda', type=float, default=10, help='arch_accecelamda')

    # model setting
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    parser.add_argument('--notrain', action='store_true', default=False, help='train 0 epoch')
    parser.add_argument('--usefocalweight', action='store_true', default=False, help='usefocalweight')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--ftlr', type=float, default=1e-4, help='fine tune learning rate')

    # gumbel setting
    parser.add_argument('--gumbel_scale', type=float, default=10e-0, help='gumbel_scale')

    # others
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--debug', action='store_true', default=False, help='set logging level to debug')
    parser.add_argument('--dataset_name', type=str, default=None)

    # GAE related
    parser.add_argument('--opt_num', type=int, default=11)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--num_mlp_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    # data
    parser.add_argument('--preprocess_mode', type=int, default=4)
    parser.add_argument('--preprocess_lamb', type=float, default=0.)

    args, unknown_args = parser.parse_known_args()

    args.save = '../checkpoints/search-{}-{}-{}-{}'.format(
        args.dataset_name, args.model_name, time.strftime("%Y%m%d-%H%M%S"), np.random.randint(100)
    )
    utils.create_exp_dir(
        path=args.save,
        scripts_to_save=glob.glob('*.py') + glob.glob('module/**/*.py', recursive=True)
    )

    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging_level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(stream=sys.stdout, level=logging_level,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    main()
