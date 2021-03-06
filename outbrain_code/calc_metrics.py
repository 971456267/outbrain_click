#thanks to http://github.com/daiwk/gt/bolb/master/tools/calc_metrics.py#L194
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt


def generate_demo_data(class_num=2, starts_from=0):
    """
    given class_num, generate demo data in np.array format
    """
    label_prefix = "class_"
    labels_name = []
    if class_num == 2:
        g_threshold = 0.31 # just test
        neg_label = starts_from
        pos_label = starts_from + 1
        labels_true = np.array([neg_label, neg_label, pos_label, pos_label])
        labels_pred_prob = np.array([0.1, 0.4, 0.35, 0.8])

        labels_pred = np.array([pos_label if i > g_threshold else neg_label for i in labels_pred_prob])
        for idx in xrange(0, 2):
            labels_name.append(label_prefix + '%d' % idx)
    else:
        labels_true = np.array([1, 2, 1, 1, 2, 3])
        labels_pred_prob = np.array([0.3, 0.24, 0.35, 0.8, 0.7, 0.33])

        labels_pred = np.array([1, 2, 3, 2, 2, 1])
        for idx in xrange(0, 3):
            labels_name.append(label_prefix + '%d' % idx)
        
    return labels_true, labels_pred_prob, labels_pred, labels_name


def get_confusion_matrix(labels_true, labels_pred, class_num, starts_from):
    """
    Args:
        labels_true
        labels_pred
    Returns:
        (confusion_matrix_res, (tn, fp, fn, tp))

    """
    confusion_matrix_res = confusion_matrix(labels_true, labels_pred)

    if class_num == 2 and starts_from == 0:

        (tn, fp, fn, tp) = confusion_matrix(labels_true, labels_pred).ravel()
        print "tn: ", tn
        print "fp: ", fp
        print "fn: ", fn
        print "tp: ", tp
    
    return confusion_matrix_res


def get_accuracy(labels_true, labels_pred, normalize_type=True):
    """
    In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
    Args:
        labels_true
        labels_pred
        normalize_type:
            return the number of correctly classified samples. Otherwise, return the fraction of correctly classified samples.
    Returns:
        accuracy
    """
    return accuracy_score(labels_true, labels_pred, normalize=normalize_type)


def get_recall(labels_true, labels_pred, average_type=None):
    """
        tp / (tp + fn)
    Args:
        labels_true
        labels_pred
        average_type:
            + 
            + micro: Calculate metrics globally by counting the total true positives, false negatives and false positives.
            + macro: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
            + weighted: Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label). This alters 'macro' to account for label imbalance; it can result in an F-score that is not between precision and recall.
            + samples: Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where this differs from accuracy_score).
            + None: the scores for each class are returned. 
    Returns:
        recall
    """
    return recall_score(labels_true, labels_pred, average=average_type)


def get_precision(labels_true, labels_pred, average_type):
    """
        tp / (tp + fp)
    Args:
        labels_true
        labels_pred
        average_type:
            + micro: Calculate metrics globally by counting the total true positives, false negatives and false positives.
            + macro: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
            + weighted: Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label). This alters 'macro' to account for label imbalance; it can result in an F-score that is not between precision and precision.
            + samples: Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where this differs from accuracy_score).
            + None: the scores for each class are returned. 
    Returns:
        precision
    """
    return precision_score(labels_true, labels_pred, average=average_type)

def get_f1(labels_true, labels_pred, average_type=None):
    """
    Args:
        labels_true
        labels_pred
        average_type:
            + micro: Calculate metrics globally by counting the total true positives, false negatives and false positives.
            + macro: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
            + weighted: Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label). This alters 'macro' to account for label imbalance; it can result in an F-score that is not between precision and recall.
            + samples: Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where this differs from accuracy_score).
            + None: the scores for each class are returned. 
    Returns:
        f1
    """
    return f1_score(labels_true, labels_pred, average=average_type)


def get_auc(labels_true, labels_pred_prob, pos_label, class_num, starts_from=0):
    """
    Args:
        labels_true
        labels_pred_prob
        pos_label: Label considered as positive and others are considered negative.
        class_num: if class_num == 2 and label starts from 0: `roc_auc_score` equals to `roc_curve then auc`' s res
    Returns:
        auc

    """
    if class_num == 2 and starts_from == 0:
        auc_res = roc_auc_score(labels_true, labels_pred_prob)
    else:
        fpr, tpr, thresholds = roc_curve(labels_true, labels_pred_prob, pos_label=pos_label)
        print "fpr", fpr
        print "tpr", tpr
        print "thresholds", thresholds

        auc_res = auc(fpr, tpr)

    fpr, tpr, thresholds = roc_curve(labels_true, labels_pred_prob, pos_label=pos_label)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='pos_label=%d' % (pos_label))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')

    plt.savefig("tmp_roc.png")

    return auc_res


if __name__ == "__main__":
    ## settings
    class_num = 2 # or 3
    starts_from = 0 # or 1
    pos_label = 1 # if not binary, it is defined by user... if binary, it is starts_from + 1

    labels_true, labels_pred_prob, labels_pred, labels_name = generate_demo_data(class_num, starts_from) 
    print "------------------------------------"
    print "labels_name", labels_name
    print "labels_true: ", labels_true
    print "labels_pred_prob: ", labels_pred_prob
    print "labels_pred: ", labels_pred
    normalize_type = True
    print "class_num: ", class_num

    print "------------------------------------"

    ## confusion_matrix 
    print "confusion matrix: "
    confusion_matrix_res = get_confusion_matrix(labels_true, labels_pred, class_num, starts_from)
    print confusion_matrix_res
    print "------------------------------------"

    ## accuracy
    print "accuracy: ", get_accuracy(labels_true, labels_pred, normalize_type)

    print "-------- average_type: micro -------"

    ## micro recall/precision/f1
    average_type = "micro"
    print "recall: ", get_recall(labels_true, labels_pred, average_type)
    print "precision: ", get_precision(labels_true, labels_pred, average_type)
    print "f1: ", get_f1(labels_true, labels_pred, average_type)

    print "-------- average_type: None --------"

    ## None recall/precision/f1
    average_type = None
    print "recall: ", get_recall(labels_true, labels_pred, average_type)
    print "precision: ", get_precision(labels_true, labels_pred, average_type)
    print "f1: ", get_f1(labels_true, labels_pred, average_type)

    print "------------------------------------"

    ## auc
    print "auc: " 
    print get_auc(labels_true, labels_pred_prob, pos_label, class_num, starts_from)

    print "------------------------------------"

    ## classification report
    print "classification report: "
    print classification_report(labels_true, labels_pred, target_names=labels_name)


