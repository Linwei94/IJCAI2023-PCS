import os
import sys
import torch
import random
import argparse
from torch import nn
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from train_utils import train_single_epoch, test_single_epoch

# Import dataloaders
import data.cifar10 as cifar10
import data.cifar100 as cifar100
import data.tiny_imagenet as tiny_imagenet

# Import network architectures
from module.resnet_tiny_imagenet import resnet50 as resnet50_ti
from module.resnet import resnet50, resnet110
from module.wide_resnet import wide_resnet_cifar
from module.densenet import densenet121

# Import metrics to compute
from metrics.metrics import test_classification_net_logits
from metrics.metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss
from metrics.metrics import maximum_calibration_error

# Import temperature scaling and NLL utilities
from temperature_scaling import ModelWithTemperature

from metrics.plots import reliability_plot, bin_strength_plot, ax_reliability_plot, ax_bin_strength_plot
from metrics.metrics import test_classification_net

import wandb






def parseArgs():
    default_dataset = 'cifar10'
    dataset_root = './'
    model = 'resnet50'
    save_loc = './'
    num_bins = 15
    model_name = None
    train_batch_size = 128
    test_batch_size = 128
    cross_validation_error = 'ece'

    parser = argparse.ArgumentParser(
        description="Evaluating a single model on calibration metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        dest="dataset", help='dataset to test on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--model-name", type=str, default=model_name,
                        dest="model_name", help='name of the model')
    parser.add_argument("--model", type=str, default=model, dest="model",
                        help='Model to test')
    parser.add_argument("--save-path", type=str, default=save_loc,
                        dest="save_loc",
                        help='Path to import the model')
    parser.add_argument("--num-bins", type=int, default=num_bins, dest="num_bins",
                        help='Number of bins')
    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("-da", action="store_true", dest="data_aug",
                        help="Using data augmentation")
    parser.set_defaults(data_aug=True)
    parser.add_argument("-b", type=int, default=train_batch_size,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("-tb", type=int, default=test_batch_size,
                        dest="test_batch_size", help="Test Batch size")
    parser.add_argument("--test_block_index", type=int, default=0,
                        dest="test_block_index")
    parser.add_argument("--cverror", type=str, default=cross_validation_error,
                        dest="cross_validation_error", help='Error function to do temp scaling')
    parser.add_argument("-log", action="store_true", dest="log",
                        help="whether to print log data")

    # test combination
    parser.add_argument("--combination", type=str, default="186,313,299,139,189")
    parser.add_argument("--tune_epoch", type=int, default=1)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--lr", type=float, default=1e-4)

    return parser.parse_args()

def test_performance():
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = ECELoss().cuda()
    adaece_criterion = AdaptiveECELoss().cuda()
    cece_criterion = ClasswiseECELoss().cuda()

    logits, labels = get_logits_labels(test_loader, net)
    conf_matrix, p_accuracy, out_labels, predictions, confidences = test_classification_net_logits(logits,
                                                                                                    labels)

    p_ece = ece_criterion(logits, labels).item()
    p_adaece = adaece_criterion(logits, labels).item()
    p_cece = cece_criterion(logits, labels).item()
    p_nll = nll_criterion(logits, labels).item()
    p_mce = maximum_calibration_error(confidences, predictions, out_labels, num_bins=num_bins)

    res_str = '{:s}&{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}'.format(model_name, p_accuracy, p_nll, p_ece,
                                                                        p_adaece, p_cece, p_mce)
    print("-----------------Pre T-----------------")
    print("p_accuracy:", p_accuracy)
    print("p_ece:", p_ece)
    print("p_adaece:", p_adaece)
    print("p_cece:", p_cece)
    print("p_nll:", p_nll)
    print("p_mce:", p_mce)

    scaled_model = ModelWithTemperature(net, args.log)
    scaled_model.set_temperature(val_loader, cross_validate=cross_validation_error)
    T_opt = scaled_model.get_temperature()
    logits, labels = get_logits_labels(test_loader, scaled_model)
    conf_matrix, accuracy, out_labels, predictions, confidences = test_classification_net_logits(logits, labels)

    ece = ece_criterion(logits, labels).item()
    adaece = adaece_criterion(logits, labels).item()
    cece = cece_criterion(logits, labels).item()
    nll = nll_criterion(logits, labels).item()
    mce = maximum_calibration_error(confidences, predictions, out_labels, num_bins=num_bins)

    res_str += '&{:.4f}({:.2f})&{:.4f}&{:.4f}&{:.4f}&{:.4f}'.format(nll, T_opt, ece, adaece, cece, mce)
    print("-----------------Post T-----------------")
    print("ece:", ece)
    print("adaece:", adaece)
    print("cece:", cece)
    print("nll:", nll)
    print("mce:", mce)

    return p_ece, p_accuracy, p_nll, ece, nll


def get_logits_labels(data_loader, net):
    logits_list = []
    labels_list = []
    net.eval()
    with torch.no_grad():
        for data, label in data_loader:
            data = data.cuda()
            logits = net(data)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
    return logits, labels


if __name__ == "__main__":
    # Checking if GPU is available
    cuda = False
    if (torch.cuda.is_available()):
        cuda = True

    # Setting additional parameters
    device = torch.device("cuda" if cuda else "cpu")

    args = parseArgs()
    args.seed = 1
    torch.manual_seed(args.seed)

    config = args
    config.experment_name = "test block overfitting"
    wandb.init(project="Predecessor-Combination-Search", config=config)

    if args.model_name is None:
        args.model_name = args.model

    dataset = args.dataset
    dataset_root = args.dataset_root
    model_name = args.model_name
    save_loc = args.save_loc
    num_bins = args.num_bins
    cross_validation_error = args.cross_validation_error

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

    # Taking input for the dataset
    num_classes = dataset_num_classes[dataset]
    if (args.dataset == 'tiny_imagenet'):
        train_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='train',
            batch_size=args.train_batch_size,
            pin_memory=args.gpu)

        val_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='val',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu)

        test_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='val',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu)
    else:
        train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            random_seed=1,
            pin_memory=args.gpu,

        )

        test_loader = dataset_loader[args.dataset].get_test_loader(
            batch_size=args.test_batch_size,
            pin_memory=args.gpu
        )

    model = models[model_name]

    net = model(num_classes=num_classes, temp=1.0)
    net.cuda()
    cudnn.benchmark = True




    # Dataset params
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
        'densenet121': densenet121
    }

    combination = [int(i) for i in args.combination.split(',')]
    args.weight_folder = "../weights/{}/{}".format(args.dataset, args.model)

    # setup model
    model = models[model_name]

    net = model(num_classes=num_classes, temp=1.0)
    net.cuda()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False
    )

    net.load_combination_weight(combination, args.weight_folder, model_name)


    print("-" * 25, "Fine-tune {} epoch using combination={} on {} and {}".format(args.tune_epoch, combination, args.dataset, args.model), "-" * 25)

    for j in range(args.tune_epoch):
        train_loss = train_single_epoch(args.tune_epoch,
                                        net,
                                        train_loader,
                                        optimizer,
                                        device)
        pre_ece, pre_acc, pre_nll, post_ece, post_nll = test_performance()

        print(
            "Epoch: {:d} \t Train Loss: {:.4f} \t Pre ECE: {:.4f} \t Pre Acc: {:.4f} \t Pre NLL: {:.4f} \t Post ECE: {:.4f} \t Post NLL: {:.4f}".format(
                j, train_loss, pre_ece, pre_acc, pre_nll, post_ece, post_nll)
        )

        wandb.log({
            "train_{}_pre_ece".format(j+1): pre_ece,
            "train_{}_pre_acc".format(j+1): pre_acc,
            "train_{}_pre_nll".format(j+1): pre_nll,
            "train_{}_post_ece".format(j+1): post_ece,
            "train_{}_post_nll".format(j+1): post_nll,
        })

    # PCS
    PCS_conf_matrix, PCS_accuracy, PCS_labels, PCS_predictions, PCS_confidences, _ = test_classification_net(net, test_loader, device)


    # test pretrained weight
    # Cross Entropy
    net = resnet50(num_classes=10, temp=1.0)
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.load_state_dict(torch.load("./pretrained_weight/resnet50_cross_entropy_350.model"))
    CE_conf_matrix, CE_accuracy, CE_labels, CE_predictions, CE_confidences, _ = test_classification_net(net, test_loader, device)

    # Label Smoothing
    net.load_state_dict(torch.load("./pretrained_weight/resnet50_cross_entropy_smoothed_smoothing_0.05_350.model"))
    LS_conf_matrix, LS_accuracy, LS_labels, LS_predictions, LS_confidences, _ = test_classification_net(net, test_loader, device)


    # FLSD-53
    net.load_state_dict(torch.load("./pretrained_weight/resnet50_focal_loss_adaptive_53_350.model"))
    FLSD_53_conf_matrix, FLSD_53_accuracy, FLSD_53_labels, FLSD_53_predictions, FLSD_53_confidences, _ = test_classification_net(net, test_loader, device)

    # FL-3
    net.load_state_dict(torch.load("./pretrained_weight/resnet50_focal_loss_gamma_3.0_350.model"))
    FL_3_conf_matrix, FL_3_accuracy, FL_3_labels, FL_3_predictions, FL_3_confidences, _ = test_classification_net(net, test_loader, device)

    # MMCE
    net.load_state_dict(torch.load("./pretrained_weight/resnet50_mmce_weighted_lamda_2.0_350.model"))
    MMCE_conf_matrix, MMCE_accuracy, MMCE_labels, MMCE_predictions, MMCE_confidences, _ = test_classification_net(net, test_loader, device)


    # full plot
    fig, axs = plt.subplots(3, 4, figsize=(40, 24), sharex=True,)
    ax_reliability_plot(axs[0][0], PCS_confidences, PCS_predictions, PCS_labels, num_bins=num_bins, title="PCS")
    ax_bin_strength_plot(axs[0][1], PCS_confidences, PCS_predictions, PCS_labels, num_bins=num_bins, title="PCS")
    ax_reliability_plot(axs[0][2], CE_confidences, CE_predictions, CE_labels, num_bins=num_bins, title="Cross Entropy")
    ax_bin_strength_plot(axs[0][3], CE_confidences, CE_predictions, CE_labels, num_bins=num_bins, title="Cross Entropy")
    ax_reliability_plot(axs[1][0], LS_confidences, LS_predictions, LS_labels, num_bins=num_bins, title="Label Smoothing")
    ax_bin_strength_plot(axs[1][1], LS_confidences, LS_predictions, LS_labels, num_bins=num_bins, title="Label Smoothing")
    ax_reliability_plot(axs[1][2], FLSD_53_confidences, FLSD_53_predictions, FLSD_53_labels, num_bins=num_bins, title="FLSD-53")
    ax_bin_strength_plot(axs[1][3], FLSD_53_confidences, FLSD_53_predictions, FLSD_53_labels, num_bins=num_bins, title="FLSD-53")
    ax_reliability_plot(axs[2][0], FL_3_confidences, FL_3_predictions, FL_3_labels, num_bins=num_bins, title="FL-3")
    ax_bin_strength_plot(axs[2][1], FL_3_confidences, FL_3_predictions, FL_3_labels, num_bins=num_bins, title="FL-3")
    ax_reliability_plot(axs[2][2], MMCE_confidences, MMCE_predictions, MMCE_labels, num_bins=num_bins, title="MMCE")
    ax_bin_strength_plot(axs[2][3], MMCE_confidences, MMCE_predictions, MMCE_labels, num_bins=num_bins, title="MMCE")

    fig.tight_layout()
    fig.show()
    fig.savefig('./experiments/reliability_plot/full1.pdf')