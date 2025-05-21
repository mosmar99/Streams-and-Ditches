# workspace/utils/metrics.py
import torch

def calculate_tp_fp_fn_per_class(preds_flat, targets_flat, num_classes):
    """
    Calculates True Positives, False Positives, and False Negatives per class.

    Args:
        preds_flat (torch.Tensor): Flattened tensor of predicted class indices (N*H*W,).
        targets_flat (torch.Tensor): Flattened tensor of ground truth class indices (N*H*W,).
        num_classes (int): Total number of classes.

    Returns:
        tuple: (tps, fps, fns)
            tps (list): List of True Positives for each class.
            fps (list): List of False Positives for each class.
            fns (list): List of False Negatives for each class.
    """
    tps = []
    fps = []
    fns = []

    for c in range(num_classes):
        true_class_c = (targets_flat == c)
        pred_class_c = (preds_flat == c)

        tp = (pred_class_c & true_class_c).sum().item()
        fp = (pred_class_c & ~true_class_c).sum().item()
        fn = (~pred_class_c & true_class_c).sum().item()
        
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        
    return tps, fps, fns

def calculate_precision_recall_per_class(tps, fps, fns, epsilon=1e-7):
    """
    Calculates precision and recall per class.

    Args:
        tps (list): List of True Positives for each class.
        fps (list): List of False Positives for each class.
        fns (list): List of False Negatives for each class.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        tuple: (precisions, recalls)
            precisions (list): List of precision scores for each class.
            recalls (list): List of recall scores for each class.
    """
    num_classes = len(tps)
    precisions = []
    recalls = []

    for c in range(num_classes):
        precision_c = tps[c] / (tps[c] + fps[c] + epsilon)
        recall_c = tps[c] / (tps[c] + fns[c] + epsilon)
        precisions.append(precision_c)
        recalls.append(recall_c)
        
    return precisions, recalls