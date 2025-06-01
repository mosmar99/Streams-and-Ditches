import torch
import math

def calculate_tp_fp_fn_tn_per_class(preds_flat, targets_flat, num_classes):
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
    tns = []

    for c in range(num_classes):
        true_class_c = (targets_flat == c)
        pred_class_c = (preds_flat == c)

        tp = (pred_class_c & true_class_c).sum().item()
        fp = (pred_class_c & ~true_class_c).sum().item()
        fn = (~pred_class_c & true_class_c).sum().item()
        tn = (~pred_class_c & ~true_class_c).sum().item()
        
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        tns.append(tn)
        
    return tps, fps, fns, tns

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

def calculate_iou_per_class(tps, fps, fns, epsilon=1e-7):
    """
    Calculates Intersection over Union (IoU) or Jaccard Index per class.

    Args:
        tps (list): List of True Positives for each class.
        fps (list): List of False Positives for each class.
        fns (list): List of False Negatives for each class.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        list: ious_per_class - List of IoU scores for each class.
    """
    num_classes = len(tps)
    ious_per_class = []

    for c in range(num_classes):
        intersection = tps[c]
        union = tps[c] + fps[c] + fns[c]
        iou_c = intersection / (union + epsilon)
        ious_per_class.append(iou_c)
        
    return ious_per_class

def calculate_dice_per_class(tps, fps, fns, epsilon=1e-7):
    """
    Calculates Dice Coefficient (F1 Score) per class.

    Args:
        tps (list): List of True Positives for each class.
        fps (list): List of False Positives for each class.
        fns (list): List of False Negatives for each class.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        list: dice_scores_per_class - List of Dice scores for each class.
    """
    num_classes = len(tps)
    dice_scores_per_class = []

    for c in range(num_classes):
        dice_c = (2 * tps[c]) / (2 * tps[c] + fps[c] + fns[c] + epsilon)
        dice_scores_per_class.append(dice_c)
        
    return dice_scores_per_class

def calculate_overall_pixel_accuracy(preds_flat, targets_flat, epsilon=1e-7):
    """
    Calculates overall pixel accuracy.

    Args:
        preds_flat (torch.Tensor): Flattened tensor of predicted class indices.
        targets_flat (torch.Tensor): Flattened tensor of ground truth class indices.
        epsilon (float): Small value to avoid division by zero if total_pixels is 0 (unlikely).

    Returns:
        float: Overall pixel accuracy.
    """
    correct_pixels = (preds_flat == targets_flat).sum().item()
    total_pixels = targets_flat.numel()
    accuracy = correct_pixels / (total_pixels + epsilon)
    return accuracy

def calculate_mcc_per_class(tps, fps, fns, tns, epsilon=1e-7):
    """
    Calculates Matthews Correlation Coefficient (MCC) per class.

    MCC is a robust metric that considers all four confusion matrix values (TP, TN, FP, FN)
    and is generally regarded as a balanced measure which can be used even if the classes
    are of very different sizes.

    Formula: (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    Args:
        tps (list): List of True Positives for each class.
        fps (list): List of False Positives for each class.
        fns (list): List of False Negatives for each class.
        tns (list): List of True Negatives for each class.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        list: mccs_per_class - List of MCC scores for each class.
    """
    num_classes = len(tps)
    mccs_per_class = []

    for c in range(num_classes):
        tp = tps[c]
        fp = fps[c]
        fn = fns[c]
        tn = tns[c]

        numerator = (tp * tn) - (fp * fn)
        
        # Denominator terms
        # Use float for these products to avoid potential overflow before sqrt
        denom_term1 = float(tp + fp)
        denom_term2 = float(tp + fn)
        denom_term3 = float(tn + fp)
        denom_term4 = float(tn + fn)
        
        denominator_product = denom_term1 * denom_term2 * denom_term3 * denom_term4

        # Calculate MCC
        # If the denominator is zero (due to zero sums in terms), MCC is typically 0.
        # This occurs when one of the marginals (e.g., actual positives, actual negatives,
        # predicted positives, predicted negatives) is zero.
        if denominator_product == 0:
            mcc_c = 0.0
        else:
            mcc_c = numerator / (math.sqrt(denominator_product) + epsilon)
        
        mccs_per_class.append(mcc_c)
        
    return mccs_per_class