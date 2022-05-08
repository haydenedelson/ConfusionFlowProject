import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.figure import SubplotParams
from plot_confusion_matrix import plot_cm_data
from plot_precision_recall import plot_pr_data
from plot_accuracy import plot_acc
from utils import get_title

def create_figure():
    """
    Create figures for plots & subplots
    """
    fig = plt.figure(figsize=(30, 12), subplotpars=SubplotParams(left=0, right=1, bottom=0, top=1))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.02)
    conf_fig, l_fig = fig.subfigures(1, 2)
    pr_fig, acc_fig = l_fig.subfigures(2, 1)
    
    return conf_fig, pr_fig, acc_fig

def plot_data(cm, fpr, fnr, acc, prec, rec, f1, num_classes, classes, class_freq, model_names):
    conf_fig, pr_fig, acc_fig = create_figure()
    plot_cm_data(conf_fig, cm, fpr, fnr, acc, num_classes, classes)
    acc_ax = plot_acc(acc_fig, acc, model_names)
    plot_pr_data(pr_fig, prec, rec, f1, num_classes, classes, class_freq)
    return conf_fig, pr_fig, acc_ax

