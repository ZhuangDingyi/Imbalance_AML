from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
import numpy as np
def check_edge_type(flag1,flag2):
    if flag1 and flag2:
        # Internal transaction
        return 0
    elif flag1 and not flag2:
        # Internal -> External
        return 1
    elif not flag1 and flag2:
        # External -> Internal
        return 2
    else:
        # External -> External, which should be deleted as we don't have the information
        return -1
    
def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate various metrics for multi-class classification.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    y_prob (array-like, optional): Predicted probabilities for each class.

    Returns:
    dict: A dictionary containing the calculated metrics.
    """
    metrics = {}
    
    # Macro and Micro F1 scores
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro')
    
    # Precision and Recall
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    if y_prob is not None:
        # AUC and Precision-Recall AUC
        metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        metrics['pr_auc'] = average_precision_score(y_true, y_prob, average='macro')
    
    return metrics