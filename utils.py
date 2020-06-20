from argparse import ArgumentParser
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import time
import scikitplot as skplt
import matplotlib.pyplot as plt
import os
import glob

import torch
import torch.nn as nn

from torchtext import data
from torchtext import datasets


def plot_confusion_matrices(model_name, epoch_num, train_y_truth, train_y_pred, validate_y_truth, validate_y_pred):
    plot_folder = "plots/models/" + model_name + "/"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    # Train CM
    skplt.metrics.plot_confusion_matrix(
        train_y_truth, train_y_pred, normalize=True)
    plt.savefig(plot_folder + epoch_num + '_train_cm.png')
    plt.title(model_name+"_train_cm")
    plt.clf()
    plt.close()

    # Validation CM
    skplt.metrics.plot_confusion_matrix(
        validate_y_truth, validate_y_pred, normalize=True)
    plt.savefig(plot_folder + epoch_num + '_valid_cm.png')
    plt.title(model_name+"_valid_cm")
    plt.clf()
    plt.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def mask_(matrices, maskval=0.0, mask_diagonal=True):
    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval
