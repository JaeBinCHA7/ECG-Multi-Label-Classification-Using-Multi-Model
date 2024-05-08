import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, classification_report, confusion_matrix


def compute_auc(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k] == 1)
        tn[0] = np.sum(labels[:, k] == 0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        tnr = np.zeros(num_thresholds)
        ppv = np.zeros(num_thresholds)
        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float('nan')
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float('nan')
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float('nan')

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds - 1):
            auroc[k] += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
            auprc[k] += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    # Compute macro AUROC and macro AUPRC across classes.
    if np.any(np.isfinite(auroc)):
        macro_auroc = np.nanmean(auroc)
    else:
        macro_auroc = float('nan')
    if np.any(np.isfinite(auprc)):
        macro_auprc = np.nanmean(auprc)
    else:
        macro_auprc = float('nan')

    return macro_auroc, macro_auprc, auroc, auprc


def compute_kappa(labels, outputs, num_classes=26):
    avg_kappa = np.zeros(num_classes)
    for class_num in range(num_classes):
        current_t_all = labels[:, class_num]
        current_binary_outputs = outputs[:, class_num]
        kappa = cohen_kappa_score(current_t_all, current_binary_outputs)
        if kappa:
            avg_kappa[class_num] = kappa
        else:
            avg_kappa[class_num] = float('nan')
    if np.any(np.isfinite(avg_kappa)):
        avg_kappa = np.nanmean(avg_kappa)
    else:
        avg_kappa = float('nan')
    return avg_kappa


# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def compute_confusion_matrices(labels, outputs, normalize=False):
    # Compute a binary confusion matrix for each class k:
    #
    #     [TN_k FN_k]
    #     [FP_k TP_k]
    #
    # If the normalize variable is set to true, then normalize the contributions
    # to the confusion matrix by the number of labels per recording.
    num_recordings, num_classes = np.shape(labels)

    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                    A[j, 1, 1] += 1
                elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                    A[j, 1, 0] += 1
                elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                    A[j, 0, 1] += 1
                elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                    A[j, 0, 0] += 1
                else:  # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                    A[j, 1, 1] += 1.0 / normalization
                elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                    A[j, 1, 0] += 1.0 / normalization
                elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                    A[j, 0, 1] += 1.0 / normalization
                elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                    A[j, 0, 0] += 1.0 / normalization
                else:  # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')

    return A


def compute_f_measure_and_precision_recall_specificity(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs)

    f_measure = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    specificity = np.zeros(num_classes)  # Specificity 추가

    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]

        if 2 * tp + fp + fn:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float('nan')

        if tp + fp:
            precision[k] = float(tp) / float(tp + fp)
        else:
            precision[k] = float('nan')

        if tp + fn:
            recall[k] = float(tp) / float(tp + fn)
        else:
            recall[k] = float('nan')

        # Specificity 계산 추가
        if tn + fp:
            specificity[k] = float(tn) / float(tn + fp)
        else:
            specificity[k] = float('nan')

    if np.any(np.isfinite(f_measure)):
        macro_f_measure = np.nanmean(f_measure)
    else:
        macro_f_measure = float('nan')

    if np.any(np.isfinite(precision)):
        macro_precision = np.nanmean(precision)
    else:
        macro_precision = float('nan')

    if np.any(np.isfinite(recall)):
        macro_recall = np.nanmean(recall)
    else:
        macro_recall = float('nan')

    # Macro Specificity 계산 추가
    if np.any(np.isfinite(specificity)):
        macro_specificity = np.nanmean(specificity)
    else:
        macro_specificity = float('nan')

    return macro_f_measure, f_measure, macro_precision, precision, macro_recall, recall, macro_specificity, specificity


def evaluate(labels, scores, res_th=None):
    '''
    metric for auc/ap
    :param labels:
    :param scores:
    :param res_th:
    :param saveto:
    :return:
    '''
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # True/False Positive Rates.
    fpr, tpr, ths = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # best f1
    best_f1 = 0
    best_threshold = 0
    for threshold in ths:
        tmp_scores = scores.copy()
        tmp_scores[tmp_scores >= threshold] = 1
        tmp_scores[tmp_scores < threshold] = 0
        cur_f1 = f1_score(labels, tmp_scores)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_threshold = threshold

    # threshold f1
    if res_th is not None:
        tmp_scores = scores.copy()
        tmp_scores[tmp_scores >= res_th] = 1
        tmp_scores[tmp_scores < res_th] = 0
        print(classification_report(labels, tmp_scores))
        print(confusion_matrix(labels, tmp_scores))
    auc_prc = average_precision_score(labels, scores)
    return auc_prc, roc_auc, best_threshold, best_f1
