import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def create_acc_axes(acc_fig):
    """
    Create axes for accuracy/zoom plot
    """
    ax = acc_fig.add_axes([0.05, 0.05, 0.9, 0.85])
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    return ax

def plot_acc(acc_fig, acc, model_names):
    """
    Plot accuracy data on zoom plot
    """
    ax = create_acc_axes(acc_fig)
    
    epochs = len(acc[0])
    for i in range(len(acc)):
        accuracy = acc[i]
        ax.plot(list(range(1, epochs + 1)), accuracy)
    ax.legend(model_names)
    ax.set_title('Overall Accuracy')
    return ax