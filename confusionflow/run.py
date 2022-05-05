import sys
import numpy as np
import tensorflow as tf
from metrics import compute_metrics
from utils import import_data, get_title
from visualize import plot_data

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Import data from log & config files
num_classes, classes, class_freq, acc, logs, model_names = import_data(sys.argv)

# Compute data to plot
cm, fnr, fpr, prec, rec, f1 = compute_metrics(logs, classes, num_classes, class_freq)

# Plot data
conf_fig, pr_fig, acc_ax = plot_data(cm, fpr, fnr, acc, prec, rec, f1, num_classes, classes, class_freq, model_names)

# Define interaction function
def on_click(event):
    axes = event.inaxes
    if axes is None:
        return
    
    axes.get_lines()

    # Get data from clicked axes
    x_vals = [line.get_xdata() for line in axes.get_lines()]
    y_vals = [line.get_ydata() for line in axes.get_lines()]
    
    # Get title for new plot
    title = get_title(axes, classes)
    
    # Plot data on zoom plot
    if len(x_vals) > 0:
        acc_ax.clear()
        for i in range(len(x_vals)):
            acc_ax.plot(x_vals[i], y_vals[i])
            acc_ax.set_title(title)
            acc_ax.set_ylim(0, 1)
            acc_ax.legend(model_names)

        axes.figure.canvas.draw()
        plt.pause(0.5)

# Bind interactions
conf_fig.canvas.mpl_connect('button_press_event', on_click)
pr_fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
