import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def create_cm_axes(cm_fig, num_classes):
    """
    Create confusion matrix subplots
    """
    axs = cm_fig.subplots(num_classes + 1, num_classes + 1, sharex=True, sharey=True, 
                          gridspec_kw = {'wspace':0.0, 'hspace':0.0})
    return axs

def label_cm_subplots(axs, classes, num_classes):
    """
    Label & style confusion matrix subplots
    """
    for i in range(num_classes + 1):
        for j in range(num_classes + 1):
            if i == 0:
                if j < num_classes:
                    axs[i, j].set(xlabel=classes[j])
                else:
                    axs[i, j].set(xlabel='FN')
                axs[i, j].xaxis.set_label_position('top')
            if j == 0:
                if i < num_classes:
                    axs[i, j].set(ylabel=classes[i])
                else:
                    axs[i, j].set(ylabel='FP')
            if i == num_classes:
                bbox = axs[i, j].get_position()
                bbox.y0 -= 0.01
                bbox.y1 -= 0.01
                axs[i, j].set_position(bbox)
            if j == num_classes:
                bbox = axs[i, j].get_position()
                bbox.x0 += 0.01
                bbox.x1 += 0.01
                axs[i, j].set_position(bbox)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    return axs   

def plot_cm_data(cm_fig, cm, fpr, fnr, acc, num_classes, classes):
    """
    Plot confusion data given confusion matrix figure
    """
    axs = create_cm_axes(cm_fig, num_classes)
    axs = label_cm_subplots(axs, classes, num_classes)
    
    models = len(cm)
    epochs = max([len(m) for m in acc])
    
    for i in range(models):
        for j in range(num_classes + 1):
            for k in range(num_classes + 1):
                # Plot nothing
                if j == k and j < num_classes and k < num_classes:
                    if i < 1:
                        axs[j, k].text(0.5, 0.5, classes[j], horizontalalignment='center',
                                       verticalalignment='center', transform=axs[j, k].transAxes)
                # Plot confusion
                elif j < num_classes and k < num_classes:
                    confusion = cm[i][j][k]
                    axs[j, k].plot(list(range(1, epochs + 1)), confusion)
                    axs[j, k].set_label({'metric': 'confusion', 'gt': j, 'pred': k})
                # Plot FN
                elif j < num_classes and k == num_classes:
                    false_neg = fnr[i][j]
                    axs[j, k].plot(list(range(1, epochs + 1)), false_neg)
                    axs[j, k].set_label({'metric': 'false negative', 'gt': j})
                # Plot FP
                elif j == num_classes and k < num_classes:
                    false_pos = fpr[i][k]
                    axs[j, k].plot(list(range(1, epochs + 1)), false_pos)
                    axs[j, k].set_label({'metric': 'false positive', 'gt': k})
                # Plot accuracy
                else:
                    accuracy = acc[i]
                    axs[j, k].plot(list(range(1, epochs + 1)), accuracy)
                    axs[j, k].set_label({'metric': 'accuracy'})