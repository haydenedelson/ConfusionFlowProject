import numpy as np
from utils import max_class

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def create_pr_axes(pr_fig, num_classes):
    """
    Create precision/recall subplots
    """
    axs = pr_fig.subplots(num_classes, 5, sharex=True, sharey=True,
                          gridspec_kw = {'wspace':0, 'hspace':0, 'width_ratios': [1, 2, 2, 2, 1]})
    pr_fig.subplots_adjust(top=0.925)
    return axs

def label_pr_subplots(axs, classes):
    """
    Label & style precision/recall subplots
    """
    for i in range(len(classes)):
        for j in range(5):
            if i == 0:
                if j == 0:
                    axs[i, j].set(xlabel='Class Labels')
                elif j == 1:
                    axs[i, j].set(xlabel='Precision')
                elif j == 2:
                    axs[i, j].set(xlabel='Recall')
                elif j == 3:
                    axs[i, j].set(xlabel='F1 Score')
                else:
                    axs[i, j].set(xlabel='Class Size')
                axs[i, j].xaxis.set_label_position('top')
            if j == 0:
                axs[i, j].text(0.5, 0.5, classes[i], horizontalalignment='center',
                               verticalalignment='center', transform=axs[i, j].transAxes)
            axs[i, j].spines['left'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    return axs

def plot_pr_data(pr_fig, prec, rec, f1, num_classes, classes, class_freq):
    """
    Plot precision/recall data given precision/recall figure
    """
    axs = create_pr_axes(pr_fig, num_classes)
    axs = label_pr_subplots(axs, classes)
    
    models = len(prec)
    epochs = max([len(m[0]) for m in prec])
    max_freq = max_class(class_freq)
    for i in range(models):
        for j in range(num_classes):
            for k in range(1, 5):
                # Plot precision
                if k == 1:
                    precision = prec[i][j]
                    axs[j, k].plot(list(range(1, epochs + 1)), precision)
                    axs[j, k].set_label({'metric': 'precision', 'gt': j})
                # Plot recall
                elif k == 2:
                    recall = rec[i][j]
                    axs[j, k].plot(list(range(1, epochs + 1)), recall)
                    axs[j, k].set_label({'metric': 'recall', 'gt': j})
                # Plot F1
                elif k == 3:
                    f1_score = f1[i][j]
                    axs[j, k].plot(list(range(1, epochs + 1)), f1_score)
                    axs[j, k].set_label({'metric': 'f1', 'gt': j})
                else:
                    axs[j, k].bar(9 * i, class_freq[i][classes[j]] / max_freq, width=7)
    return axs