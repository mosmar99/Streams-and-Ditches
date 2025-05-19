import numpy as np
import torch

class BaseClassificationMetric:
    '''Base class for common classification metric logic'''
    def __init__(self, classes, from_logits=True):
        self.from_logits = from_logits
        self._classes = list(classes) # Ensure it's a list, TP, FP, FN, TN stored per class
        self._tp = {cls: 0 for cls in self._classes}
        self._fp = {cls: 0 for cls in self._classes}
        self._fn = {cls: 0 for cls in self._classes}
        self._tn = {cls: 0 for cls in self._classes}

    def reset(self):
        for cls in self._classes:
            self._tp[cls] = 0
            self._fp[cls] = 0
            self._fn[cls] = 0
            self._tn[cls] = 0

    def add(self, predicted_logits, target_indices):
        '''
        Add results for predicted logits and target class indices.

        Parameters
        ----------
        predicted_logits : torch.Tensor
            Tensor of predicted logits or probabilities, shape [N, num_classes].
        target_indices : torch.Tensor
            Tensor of target class indices, shape [N].
        '''
        if not torch.is_tensor(predicted_logits):
            raise TypeError("predicted_logits must be a PyTorch tensor.")
        if not torch.is_tensor(target_indices):
            raise TypeError("target_indices must be a PyTorch tensor.")
        if predicted_logits.ndim != 2:
            raise ValueError(f"predicted_logits must be 2D (N, C), got {predicted_logits.ndim}D.")
        if target_indices.ndim != 1:
            raise ValueError(f"target_indices must be 1D (N), got {target_indices.ndim}D.")
        if predicted_logits.shape[0] != target_indices.shape[0]:
            raise ValueError(f"Batch size mismatch: predicted_logits {predicted_logits.shape[0]}, target_indices {target_indices.shape[0]}.")

        if self.from_logits:
            predicted_indices = predicted_logits.argmax(dim=1)
        else:
            predicted_indices = predicted_logits

        # Move to CPU and convert to numpy for easier element-wise comparison
        predicted_indices_np = predicted_indices.cpu().numpy()
        target_indices_np = target_indices.cpu().numpy()
        
        num_samples = len(target_indices_np)

        for cls_label in self._classes:
            # Create boolean arrays for the current class
            is_predicted_cls = (predicted_indices_np == cls_label)
            is_target_cls = (target_indices_np == cls_label)

            self._tp[cls_label] += np.logical_and(is_predicted_cls, is_target_cls).sum()
            self._fp[cls_label] += np.logical_and(is_predicted_cls, ~is_target_cls).sum()
            self._fn[cls_label] += np.logical_and(~is_predicted_cls, is_target_cls).sum()
            self._tn[cls_label] += np.logical_and(~is_predicted_cls, ~is_target_cls).sum()


    def labels(self, prefix=""):
        return [f"{prefix}{self.name}_{c}" for c in self._classes]

    def _get_counts(self, cls_label):
        tp = self._tp[cls_label]
        fp = self._fp[cls_label]
        fn = self._fn[cls_label]
        tn = self._tn[cls_label]
        return tp, fp, fn, tn

class Recall(BaseClassificationMetric):
    '''Class-wise Recall metric'''
    def __init__(self, classes, name="", from_logits=True):
        super().__init__(classes, from_logits=from_logits)
        self.name = name

    def value(self, prefix=""):
        results = {}
        for cls_label in self._classes:
            tp, fp, fn, tn = self._get_counts(cls_label)
            denominator = tp + fn
            if denominator > 0:
                results[f"{prefix}{self.name}_{cls_label}"] = tp / denominator
            else:
                results[f"{prefix}{self.name}_{cls_label}"] = 0.0 # Or np.nan, or handle as per specific needs
        return results

class Precision(BaseClassificationMetric): # F1 needs Precision
    '''Class-wise Precision metric'''
    def __init__(self, classes, name="", from_logits=True):
        super().__init__(classes, from_logits=from_logits)
        self.name = name

    def value(self, prefix=""):
        results = {}
        for cls_label in self._classes:
            tp, fp, fn, tn = self._get_counts(cls_label)
            denominator = tp + fp
            if denominator > 0:
                results[f"{prefix}{self.name}_{cls_label}"] = tp / denominator
            else:
                results[f"{prefix}{self.name}_{cls_label}"] = 0.0
        return results

class F1Score(BaseClassificationMetric):
    '''Class-wise F1-Score metric'''
    def __init__(self, classes, name="", from_logits=True):
        super().__init__(classes, from_logits=from_logits)
        self.name = name

    def value(self, prefix=""):
        results = {}
        for cls_label in self._classes:
            tp, fp, fn, tn = self._get_counts(cls_label)

            precision_denominator = tp + fp
            recall_denominator = tp + fn

            if precision_denominator > 0:
                precision = tp / precision_denominator
            else:
                precision = 0.0

            if recall_denominator > 0:
                recall = tp / recall_denominator
            else:
                recall = 0.0

            f1_denominator = precision + recall
            if f1_denominator > 0:
                results[f"{prefix}{self.name}_{cls_label}"] = (2 * precision * recall) / f1_denominator
            else:
                results[f"{prefix}{self.name}_{cls_label}"] = 0.0
        return results

class MCC(BaseClassificationMetric):
    '''Class-wise Matthews Correlation Coefficient (MCC) metric'''
    def __init__(self, classes, name="", from_logits=True):
        super().__init__(classes, from_logits=from_logits)
        self.name = name

    def value(self, prefix=""):
        results = {}
        for cls_label in self._classes:
            tp, fp, fn, tn = self._get_counts(cls_label)

            # Convert counts to float to prevent overflow in denominator calculation
            tp_f, fp_f, fn_f, tn_f = float(tp), float(fp), float(fn), float(tn)

            numerator = (tp_f * tn_f) - (fp_f * fn_f)
            
            d1 = tp_f + fp_f
            d2 = tp_f + fn_f
            d3 = tn_f + fp_f
            d4 = tn_f + fn_f

            denominator_squared = d1 * d2 * d3 * d4

            if denominator_squared > 0:
                mcc_val = numerator / np.sqrt(denominator_squared)
                # mcc_val = np.clip(mcc_val, -1.0, 1.0)
                results[f"{prefix}{self.name}_{cls_label}"] = mcc_val
            else:
                results[f"{prefix}{self.name}_{cls_label}"] = 0.0
        return results

class IoU(BaseClassificationMetric): # Rewriting IoU to use the same TP/FP/FN base
    '''Class-wise Intersection over Union (IoU) metric'''
    def __init__(self, classes, name="", from_logits=True):
        super().__init__(classes, from_logits=from_logits)
        self.name = name

    def value(self, prefix=""):
        results = {}
        for cls_label in self._classes:
            tp, fp, fn, _ = self._get_counts(cls_label)

            intersection = tp
            union = tp + fp + fn

            if union > 0:
                results[f"{prefix}{self.name}_{cls_label}"] = intersection / union
            else:
                results[f"{prefix}{self.name}_{cls_label}"] = 0.0
        return results
    
class MetricsList():
    '''Metrics container'''
    def __init__(self, metrics):
        self.metrics = metrics

    def add(self, predicted_logits, target_indices):
        for metric in self.metrics:
            metric.add(predicted_logits, target_indices)
    
    def value(self, prefix=""):
        merged = {}
        for metric in self.metrics:
            res = metric.value(prefix)
            merged.update(res)
        return merged
    
    def labels(self, prefix=""):
        labels = []
        for metric in self.metrics:
            labels += metric.labels(prefix)
        return labels
    
    def reset(self):
        for metric in self.metrics:
            metric.reset()


# --- Example Usage ---
if __name__ == '__main__':
    num_classes = 3
    class_labels = list(range(num_classes)) # e.g., [0, 1, 2]

    # Instantiate metrics
    recall_metric = Recall(classes=class_labels)
    precision_metric = Precision(classes=class_labels)
    f1_metric = F1Score(classes=class_labels)
    mcc_metric = MCC(classes=class_labels)
    iou_metric = IoU(classes=class_labels) # Using the new IoU

    metrics_list = [recall_metric, precision_metric, f1_metric, mcc_metric, iou_metric]

    # --- Simulated data (Batch 1) ---
    # Batch size N=4, Num classes C=3
    # Logits (raw scores from model)
    predicted_logits_batch1 = torch.tensor([
        [0.1, 0.8, 0.1],  # Pred: 1, True: 1 (TP for 1)
        [0.7, 0.2, 0.1],  # Pred: 0, True: 0 (TP for 0)
        [0.1, 0.2, 0.7],  # Pred: 2, True: 1 (FP for 2, FN for 1)
        [0.9, 0.05, 0.05] # Pred: 0, True: 2 (FP for 0, FN for 2)
    ], dtype=torch.float)
    # Target class indices
    target_indices_batch1 = torch.tensor([1, 0, 1, 2], dtype=torch.long)

    for metric in metrics_list:
        metric.add(predicted_logits_batch1, target_indices_batch1)

    print("--- After Batch 1 ---")
    for metric in metrics_list:
        print(f"{metric.__class__.__name__}: {metric.value()}")

    # --- Simulated data (Batch 2) ---
    predicted_logits_batch2 = torch.tensor([
        [0.1, 0.1, 0.8],  # Pred: 2, True: 2 (TP for 2)
        [0.6, 0.3, 0.1],  # Pred: 0, True: 0 (TP for 0)
        [0.2, 0.7, 0.1],  # Pred: 1, True: 1 (TP for 1)
        [0.2, 0.1, 0.7]   # Pred: 2, True: 2 (TP for 2)
    ], dtype=torch.float)
    target_indices_batch2 = torch.tensor([2, 0, 1, 2], dtype=torch.long)

    for metric in metrics_list:
        metric.add(predicted_logits_batch2, target_indices_batch2) # Accumulates from batch 1

    print("\n--- After Batch 2 (Accumulated) ---")
    for metric in metrics_list:
        print(f"{metric.__class__.__name__}: {metric.value()}")

    # --- Reset and process only Batch 2 ---
    print("\n--- Reset and process only Batch 2 ---")
    for metric in metrics_list:
        metric.reset()
        metric.add(predicted_logits_batch2, target_indices_batch2)
        print(f"{metric.__class__.__name__}: {metric.value()}")

    # Example of accessing labels
    print("\nMetric labels:", mcc_metric.labels())