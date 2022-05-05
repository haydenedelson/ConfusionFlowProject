import json, os, ast
import numpy as np

def create_dir(dir_path):
    """
    Create directory if none exists at the given path
    Input: directory path
    """
    dir_path = os.path.realpath(dir_path)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def read_log(log_file):
    """
    Helper function to read and store information from single callback log file
    Input: log file path
    """
    file = open(log_file)
    data = json.load(file)
    file.close()
    
    acc = data['accuracy']
    np_data = []
    for key in data:
        if key != 'accuracy':
            np_data.append(np.array(data[key]))
    return acc, np_data

def read_config(config_file):
    """
    Helper function to read and store information from single config file
    Input: config file path
    """
    file = open(config_file)
    data = json.load(file)
    file.close()
    
    return data

def import_logs(log_file_list):
    """
    Import data from log files
    Input: list of log file paths
    """
    logs_list = []
    acc_list = []
    
    for log_file in log_file_list:      
        acc, logs = read_log(log_file)
        acc_list.append(acc)
        logs_list.append(logs)
    
    return acc_list, logs_list

def import_configs(config_file_list):
    """
    Import data from config files
    Input: list of config file paths
    """
    config_list = []
    
    classes = None
    num_classes = None
    for config_file in config_file_list:
        class_freqs = read_config(config_file)
        config_list.append(class_freqs)
        
        if classes == None:
            classes = list(class_freqs.keys())
            num_classes = len(classes)
        else:
            if classes != list(class_freqs.keys()) or num_classes != len(class_freqs):
                raise ValueError("Classes must be the same across models")
    
    return num_classes, classes, config_list

def import_data(arg_list):
    """
    Import data given command line list of log files and config files
    Input: list of file paths
    """
    log_file_list = []
    config_file_list = []
    model_name_list = []
    for i in range(1, len(arg_list)):
        if i % 2 == 1:
            log_file_list.append(arg_list[i])

            basename = os.path.basename(arg_list[i])
            model_name_list.append(basename.rsplit('.', 1)[0])
        else:
            config_file_list.append(arg_list[i])
    print(model_name_list)
    
    if len(log_file_list) != len(config_file_list):
        raise ValueError("Must have same number of config files and log files")
    if len(log_file_list) > 4:
        raise ValueError("Max 4 log files")
    
    acc_logs, confusion_logs = import_logs(log_file_list)
    num_classes, classes, class_freqs = import_configs(config_file_list)
    
    return num_classes, classes, class_freqs, acc_logs, confusion_logs, model_name_list

def max_class(class_freq):
    max_freq = 0
    for classes_dict in class_freq:
        for val in classes_dict.values():
            if val > max_freq:
                max_freq = val
    return max_freq

def get_title(ax, classes):
    try:
        label = ast.literal_eval(ax.get_label())
    except:
        return
    
    metric = label['metric']
    gt = int(label['gt']) if 'gt' in label else None
    pred = int(label['pred']) if 'pred' in label else None
    
    if metric == 'confusion':
        return "% of Confused Instances for Class {} with {}".format(classes[gt], classes[pred])
    elif metric == 'false negative':
        return "False Negative Rate for All Classes Given {}".format(classes[gt])
    elif metric == 'false positive':
        return "False Positive Rate for All Classes Predicted As {}".format(classes[gt])
    elif metric == 'precision':
        return "Precision [%] for Class {}".format(classes[gt])
    elif metric == 'recall':
        return "Recall [%] for Class {}".format(classes[gt])
    elif metric == 'f1':
        return "F1 Score [%] for Class {}".format(classes[gt])
    else:
        return "Overall Accuracy"

