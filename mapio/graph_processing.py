import cv2
import numpy as np
from scipy.ndimage import center_of_mass
from skimage.segmentation import watershed
from skimage.morphology import remove_small_holes, skeletonize
from skimage import feature
import matplotlib.pyplot as plt
import time

def fill_small_segments(multi_seg_mask, min_size=10):
    kernel = np.array([[0,1,0],
                       [1,1,1],
                       [0,1,0]], dtype=np.uint8)
    # kernel = np.ones((2,2), dtype=np.uint8)
    
    for segment in np.unique(multi_seg_mask):
        if segment == 0:
            continue
        seg_mask = segment == multi_seg_mask
        pixels_in_segment = np.sum(seg_mask)
        if pixels_in_segment < min_size:
            dialated_seg_mask = cv2.dilate(seg_mask.astype(np.uint8), kernel, iterations=1).astype(np.bool8)
            neighbors = np.logical_xor(seg_mask, dialated_seg_mask)
            unique, counts = np.unique(multi_seg_mask[neighbors], return_counts=True)

            if len(unique) > 1:
                unique_bg_mask = unique != 0
                unique = unique[unique_bg_mask]
                counts = counts[unique_bg_mask]
            
            multi_seg_mask[seg_mask] = unique[np.argmax(counts)]
    return multi_seg_mask
            
def extract_majority_labels(multi_seg_mask, target):
    labels = []
    for segment in np.unique(multi_seg_mask):
        if segment == 0:
            continue
        seg_mask = segment == multi_seg_mask

        target_masked = target[seg_mask]
        unique, counts = np.unique(target_masked, return_counts=True)

        counts[unique == 0] = counts[unique == 0]//6

        majority_index = np.argmax(counts)
        labels.append(unique[majority_index])
    return np.array(labels)

def extract_mean_probabilities(multi_seg_mask, target):
    labels = []

    for segment in np.unique(multi_seg_mask):
        if segment == 0:
            continue
        seg_mask = segment == multi_seg_mask

        target_masked = target[:, seg_mask]
        prob_means = target_masked.mean(axis=1)
        labels.append(prob_means)

    return np.array(labels)

def extract_deep_features(coords, target):
    pad_size = 12//2
    index_conversion = 512//target.shape[1]
    labels = []
    for coord in coords:
        x, y = coord[0], coord[1]
        x, y = (x+pad_size)//index_conversion, (y+pad_size)//index_conversion
        segment_features = target[:,y,x]
        labels.append(segment_features)
    return np.array(labels)

def image_to_graph(image_preds, image_probs, label_image, feature_map, slope_image):
    canny = np.zeros_like(image_preds)
    for label in np.unique(image_preds):
        if label == 0:
            continue
        label_mask = image_preds == label

        # kernel = np.ones((3, 3), np.uint8)
        kernel = np.array([[0,1,0],
                           [1,1,1],
                           [0,1,0]], np.uint8)
        
        dia_target = cv2.dilate(label_mask.astype(np.uint8), kernel, iterations=1).astype(np.bool)

        canny += feature.canny(dia_target, sigma=2)
    canny = (canny != 0)
    canny = remove_small_holes(canny, area_threshold=64)
    canny = skeletonize(canny)

    ws = watershed(canny, markers=4800)

    ws_alpha = (ws + 1) * (image_preds != 0)
    ws_alpha = fill_small_segments(ws_alpha)

    labels = np.unique(ws_alpha)
    labels = labels[labels != 0]

    label_mapping = {old_label: new_label for new_label, old_label in enumerate(labels, start=1)}

    ws_alpha_reindex = np.copy(ws_alpha)
    for old_label, new_label in label_mapping.items():
        ws_alpha_reindex[ws_alpha == old_label] = new_label

    labels_reindex = [label_mapping[label] for label in labels]

    centers = center_of_mass(image_preds != 0, ws_alpha_reindex, labels_reindex)
    centers_int = np.round(centers).astype(int)

    predicted_label = extract_mean_probabilities(ws_alpha_reindex, image_probs)
    target_label = extract_majority_labels(ws_alpha_reindex, label_image)
    slope_stats = extract_slope_statistics(ws_alpha_reindex, slope_image)
    deep_features = extract_deep_features(centers_int, feature_map)
    labeled_centers = np.column_stack((np.array(labels_reindex)-1, centers_int, predicted_label, slope_stats, deep_features, target_label))

    # dilated_ws = grey_dilation(ws_alpha, size=(3, 3))

    H, W = ws_alpha_reindex.shape
    label_pairs = set()

    # 8-connectivity shifts
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dy, dx in shifts:
        # Define slicing ranges avoiding edges
        y_src = slice(max(0, -dy), H - max(0, dy))
        x_src = slice(max(0, -dx), W - max(0, dx))
        y_dst = slice(max(0, dy), H - max(0, -dy))
        x_dst = slice(max(0, dx), W - max(0, -dx))

        current = ws_alpha_reindex[y_src, x_src]
        neighbor = ws_alpha_reindex[y_dst, x_dst]

        mask = (current != neighbor) & (current != 0) & (neighbor != 0)
        pairs = np.stack([current[mask], neighbor[mask]], axis=1)

        for a, b in pairs:
            if a != b:
                label_pairs.add(tuple(sorted((a - 1, b - 1))))
    
    return np.array(labeled_centers, dtype=np.float32), np.array(list(label_pairs), dtype=np.int32), ws_alpha_reindex

def graph_to_image(node_predictions, node_mask):
    output = np.zeros_like(node_mask)
    for node_id in np.unique(node_mask):
        if node_id == 0:
            continue
        seg_mask = node_id == node_mask
        output[seg_mask] = node_predictions[node_id-1]
    return output

def plot_graph_edges(ax, nodes, edges):
    for edge in edges:
        label_a, label_b = edge
        coord_a = nodes[nodes[:, 0] == label_a][0, 1:]
        coord_b = nodes[nodes[:, 0] == label_b][0, 1:]

        ax.plot([coord_a[1], coord_b[1]], [coord_a[0], coord_b[0]], 'k-', lw=0.3)
    return ax

def extract_slope_statistics(multi_seg_mask, slope_image):
    """Calculates min, mean, max slope for each segment."""
    stats = []
    unique_segment_ids = np.unique(multi_seg_mask)
    unique_segment_ids = unique_segment_ids[unique_segment_ids != 0] 

    for segment_id in unique_segment_ids:
        # if segment_id == 0: continue

        seg_mask = (multi_seg_mask == segment_id)
        if np.any(seg_mask):
            slope_values_in_segment = slope_image[seg_mask]
            if slope_values_in_segment.size > 0:
                stats.append([
                    np.min(slope_values_in_segment),
                    np.mean(slope_values_in_segment),
                    np.max(slope_values_in_segment)
                ])
            else:
                stats.append([0.0, 0.0, 0.0])
        else:
            stats.append([0.0, 0.0, 0.0])
    return np.array(stats, dtype=np.float32)
