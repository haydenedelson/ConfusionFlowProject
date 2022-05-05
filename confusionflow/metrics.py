import numpy as np

def create_confusion_matrix(logs, classes, num_classes, class_freq):
    """
    Create multi-class confusion matrices for each epoch of each model trained
    """
    if len(logs) == 0:
        raise ValueError("No logs")
    
    output = []
    
    for i in range(len(logs)):
        l2_cm = [[[] for c in range(num_classes)] for c in range(num_classes)]
        
        curr_log = logs[i]
        for curr_cm in curr_log:
            for row in range(len(curr_cm)):
                curr_class = classes[row]
                for col in range(len(curr_cm[0])):
                    l2_cm[row][col].append(curr_cm[row][col] / class_freq[i][curr_class])
        output.append(l2_cm)
    return output

def compute_false_neg(logs, num_classes):
    """
    Compute number of false negative predictions for each epoch and 
    each class of each model trained
    """
    if len(logs) == 0:
        raise ValueError("No logs")
    
    output = []
    
    for curr_log in logs:
        fn = [[] for c in range(num_classes)]
    
        for curr_cm in curr_log:
            for row in range(len(curr_cm)):
                fn[row].append(np.sum(curr_cm[row]) - curr_cm[row][row])
        output.append(fn)
    return np.array(output, dtype='float')

def compute_false_pos(logs, num_classes):
    """
    Compute number of false positive predictions for each epoch and 
    each class of each model trained
    """
    if len(logs) == 0:
        raise ValueError("No logs")
    
    output = []
    
    for curr_log in logs:
        fp = [[] for c in range(num_classes)]

        for curr_cm in curr_log:
            for col in range(len(curr_cm[0])):
                fp[col].append(np.sum(curr_cm[:, col]) - curr_cm[col][col])
        output.append(fp)
    return np.array(output, dtype='float')

def compute_true_neg(logs, num_classes):
    """
    Compute number of true negative predictions for each epoch and 
    each class of each model trained
    """
    output = []
    
    for curr_log in logs:
        tn = [[] for c in range(num_classes)]
    
        for curr_cm in curr_log:
            for row in range(len(curr_cm)):
                tn[row].append(np.sum(curr_cm) - np.sum(curr_cm[row]) - np.sum(curr_cm[:, row]) + curr_cm[row][row])
        output.append(tn)
    return np.array(output)

def compute_false_neg_rate(logs, fn, num_classes):
    """
    Compute false negative rate for each epoch and each class of each model trained
    """
    output = []
    
    for i in range(len(logs)):
        fnr = [[] for c in range(num_classes)]
    
        curr_fn = fn[i]
        curr_log = logs[i]
        for row in range(len(curr_fn)):
            for col in range(len(curr_fn[row])):
                fnr[row].append(curr_fn[row][col] / (curr_fn[row][col] + curr_log[col][row][row]))
        output.append(fnr)
    return output

def compute_false_pos_rate(fp, tn, num_classes):
    """
    Compute false positive rate for each epoch and each class of each model trained
    """
    output = []
    
    for i in range(len(fp)):
        fpr = [[] for c in range(num_classes)]
    
        curr_fp = fp[i]
        curr_tn = tn[i]
        for row in range(len(curr_fp)):
            for col in range(len(curr_fp[row])):
                fpr[row].append(curr_fp[row][col] / (curr_fp[row][col] + curr_tn[row][col]))
        output.append(fpr)
    return output

def compute_precision(logs, fp, num_classes):
    """
    Compute precision values for each epoch and each class of each model trained
    """
    output = []
    
    for i in range(len(logs)):
        prec = [[] for c in range(num_classes)]
        
        curr_fp = fp[i]
        curr_log = logs[i]
        for row in range(len(curr_fp)):
            for col in range(len(curr_fp[row])):
                if curr_log[col][row][row] + curr_fp[row][col] == 0:
                    prec[row].append(np.nan)
                else:
                    prec[row].append(curr_log[col][row][row] / (curr_log[col][row][row] + curr_fp[row][col]))
        output.append(prec)
    return output

def compute_recall(logs, fn, num_classes):
    """
    Compute recall values for each epoch and each class of each model trained
    """
    output = []
    
    for i in range(len(logs)):
        rec = [[] for c in range(num_classes)]

        curr_fn = fn[i]
        curr_log = logs[i]
        for row in range(len(curr_fn)):
            for col in range(len(curr_fn[row])):
                if curr_log[col][row][row] + curr_fn[row][col] == 0:
                    rec[row].append(np.nan)
                else:
                    rec[row].append(curr_log[col][row][row] / (curr_log[col][row][row] + curr_fn[row][col]))
        output.append(rec)
    return output

def compute_f1(prec, rec, num_classes):
    """
    Compute F1 scores for each epoch and each class of each model trained
    """
    output = []
        
    for i in range(len(prec)):
        f1 = [[] for c in range(num_classes)]
        
        curr_prec = prec[i]
        curr_rec = rec[i]
        for row in range(len(curr_prec)):
            for col in range(len(curr_prec[row])):
                if curr_prec[row][col] + curr_rec[row][col] == 0:
                    f1[row].append(np.nan)
                else:
                    f1[row].append((2 * curr_prec[row][col] * curr_rec[row][col]) / 
                                   (curr_prec[row][col] + curr_rec[row][col]))
        output.append(f1)
    return output

def compute_metrics(logs, classes, num_classes, class_freq):
    cm = create_confusion_matrix(logs, classes, num_classes, class_freq)
    fn = compute_false_neg(logs, num_classes)
    fp = compute_false_pos(logs, num_classes)
    tn = compute_true_neg(logs, num_classes)
    fnr = compute_false_neg_rate(logs, fn, num_classes)
    fpr = compute_false_pos_rate(fp, tn, num_classes)
    prec = compute_precision(logs, fp, num_classes)
    rec = compute_recall(logs, fn, num_classes)
    f1 = compute_f1(prec, rec, num_classes)
    return cm, fnr, fpr, prec, rec, f1