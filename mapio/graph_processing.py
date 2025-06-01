import cv2
import numpy as np
from scipy.ndimage import center_of_mass, find_objects
# from skimage.segmentation import watershed
from skimage.morphology import remove_small_holes, skeletonize
from skimage import feature
import matplotlib.pyplot as plt
from time import time

# TESTING
from skimage.segmentation import slic, felzenszwalb, quickshift, watershed, mark_boundaries
from skimage.filters import sobel, meijering, sato, frangi, hessian, gaussian, threshold_sauvola
from skimage.measure import label, regionprops

def color_segments_rand(segments):
   # Generate random colors for each label
    unique_labels = np.unique(segments)  # Find all unique segment labels
    random_colors = np.array([np.random.rand(3,) for _ in unique_labels])  # Random RGB colors

    # Initialize the colored label image
    colored_labels = np.zeros((segments.shape[0], segments.shape[1], 3))

    # Assign random colors to each label
    for i, label in enumerate(unique_labels):
        colored_labels[segments == label] = random_colors[i]
    
    return colored_labels

def segments_rand(segments, seed=42):
    np.random.seed(seed)
   # Generate random colors for each label
    unique_labels = np.unique(segments)  # Find all unique segment labels
    random_colors = np.array([np.random.randint(3000,10000)/10000.0 for _ in unique_labels])  # Random RGB colors

    # Initialize the colored label image
    colored_labels = np.zeros((segments.shape[0], segments.shape[1]))

    # Assign random colors to each label
    for i, label in enumerate(unique_labels):
        if label == 0:
            colored_labels[segments == label] = 0
        else:
            colored_labels[segments == label] = random_colors[i]
    
    return colored_labels

def get_seg_bbox(segment_mask):
    ys, xs = np.where(segment_mask)

    if ys.size == 0 or xs.size == 0:
        return None  # or raise an exception if mask is empty

    min_y, max_y = ys.min(), ys.max()
    min_x, max_x = xs.min(), xs.max()

    return max_x, min_x, max_y, min_y

def remove_square_segments(multi_seg_mask, min_in_seg=0.95):
    res = np.zeros_like(multi_seg_mask)
    for segment in np.unique(multi_seg_mask):
        if segment == 0:
            continue
        seg_mask = segment == multi_seg_mask
        max_x, min_x, max_y, min_y = get_seg_bbox(seg_mask)
        height = max_y - min_y
        width = max_x - min_x
        area = height * width
        area_in_seg = np.sum(multi_seg_mask[min_y:max_y , min_x:max_x] == segment)
        contained = area_in_seg/area

        if contained < min_in_seg:
            res[seg_mask] = segment
    return res

def remove_square_segments_optimized(multi_seg_mask, min_in_seg=0.95):
    res = np.zeros_like(multi_seg_mask)
    
    if multi_seg_mask.size == 0:
        return res

    original_unique_labels_in_mask = np.unique(multi_seg_mask)

    segment_bboxes_inclusive = {}

    rows_all, cols_all = np.indices(multi_seg_mask.shape)
    
    rows_flat = rows_all.ravel()
    cols_flat = cols_all.ravel()
    
    labels_all_flat = multi_seg_mask.ravel()
    
    if labels_all_flat.size > 0:
        sort_indices = np.argsort(labels_all_flat, kind='mergesort')
        
        sorted_labels = labels_all_flat[sort_indices]
        sorted_rows = rows_flat[sort_indices]
        sorted_cols = cols_flat[sort_indices]

        unique_s_labels, group_start_indices = np.unique(sorted_labels, return_index=True)
        
        group_end_indices = np.concatenate((group_start_indices[1:], [len(sorted_labels)]))

        for i, label_id in enumerate(unique_s_labels):
            start = group_start_indices[i]
            end = group_end_indices[i]
            
            segment_group_rows = sorted_rows[start:end]
            segment_group_cols = sorted_cols[start:end]
            
            segment_bboxes_inclusive[label_id] = (
                segment_group_cols.max(), # max_x_inclusive
                segment_group_cols.min(), # min_x_inclusive
                segment_group_rows.max(), # max_y_inclusive
                segment_group_rows.min()  # min_y_inclusive
            )

    segments_to_keep_ids = []

    for segment_id in original_unique_labels_in_mask:
        if segment_id == 0: 
            continue

        bbox_coords = segment_bboxes_inclusive.get(segment_id)
        
        if bbox_coords is None:
            continue
            
        max_x_incl, min_x_incl, max_y_incl, min_y_incl = bbox_coords

        height_orig_style = max_y_incl - min_y_incl
        width_orig_style  = max_x_incl - min_x_incl
        
        area_orig_style = float(height_orig_style * width_orig_style)
        
        area_in_seg_orig_style = 0
        if height_orig_style >= 0 and width_orig_style >= 0:
            slice_y_start = min_y_incl
            slice_y_end   = max_y_incl 
            slice_x_start = min_x_incl
            slice_x_end   = max_x_incl

            if slice_y_end > slice_y_start and slice_x_end > slice_x_start:
                sub_mask_region = multi_seg_mask[slice_y_start:slice_y_end, 
                                                 slice_x_start:slice_x_end]
                area_in_seg_orig_style = np.sum(sub_mask_region == segment_id)

        if area_orig_style == 0:
            pass 
        else:
            contained_ratio = area_in_seg_orig_style / area_orig_style
            if contained_ratio < min_in_seg:
                segments_to_keep_ids.append(segment_id)
            
    if segments_to_keep_ids:
        keep_mask = np.isin(multi_seg_mask, segments_to_keep_ids)
        res[keep_mask] = multi_seg_mask[keep_mask]
        
    return res

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

        counts[unique == 0] = counts[unique == 0]//2

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

def segmentation_slic(image_probs):
    shedsl = slic(image_probs, n_segments=4800, channel_axis=0)
    shed_alpha = remove_square_segments_optimized(shedsl, min_in_seg=0.99)

    return shed_alpha

def segmentation_canny_ws(image_preds):
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
    return ws_alpha

def image_to_graph(image_preds, image_probs, label_image, feature_map_x9, feature_map_x7, feature_map_u7, slope_image, flow_acc, twi, elevation):
    # ws_alpha = segmentation_slic_all(elevation, label_image)
    ws_alpha = segmentation_slic(image_probs)

    labels = np.unique(ws_alpha)
    labels = labels[labels != 0]

    label_mapping = {old_label: new_label for new_label, old_label in enumerate(labels, start=1)}

    ws_alpha_reindex = np.copy(ws_alpha)
    for old_label, new_label in label_mapping.items():
        ws_alpha_reindex[ws_alpha == old_label] = new_label

    labels_reindex = [label_mapping[label] for label in labels]

    centers = center_of_mass(ws_alpha != 0, ws_alpha_reindex, labels_reindex)
    centers_int = np.round(centers).astype(int)

    predicted_label = extract_mean_probabilities(ws_alpha_reindex, image_probs)
    target_label = extract_majority_labels(ws_alpha_reindex, label_image)
    slope_stats = extract_slope_statistics(ws_alpha_reindex, slope_image)
    flow_twi_stats = extract_twi_flowacc_statistics(ws_alpha_reindex, twi, flow_acc)

    deep_features_x9 = extract_deep_features(centers_int, feature_map_x9)
    deep_features_x7 = extract_deep_features(centers_int, feature_map_x7)
    deep_features_u7 = extract_deep_features(centers_int, feature_map_u7)

    labeled_centers = np.column_stack((np.array(labels_reindex)-1, centers_int, predicted_label, slope_stats,
                                       flow_twi_stats, deep_features_x9, deep_features_x7, deep_features_u7, target_label))
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

        ax.plot([coord_a[1], coord_b[1]], [coord_a[0], coord_b[0]], 'k-', c="black", lw=0.7)
    return ax

def extract_slope_statistics(multi_seg_mask, slope_image):
    """Calculates min, mean, max slope for each segment."""
    stats = []
    unique_segment_ids = np.unique(multi_seg_mask)
    unique_segment_ids = unique_segment_ids[unique_segment_ids != 0] 

    for segment_id in unique_segment_ids:

        seg_mask = (multi_seg_mask == segment_id)
        slope_image_squeezed = slope_image.squeeze(0)
        slope_values_in_segment = slope_image_squeezed[seg_mask]

        stats.append([
            np.min(slope_values_in_segment),
            np.mean(slope_values_in_segment),
            np.max(slope_values_in_segment),
            np.std(slope_values_in_segment),
            np.sum(seg_mask)
        ])

    return np.array(stats, dtype=np.float32)

def extract_twi_flowacc_statistics(multi_seg_mask, twi, flowacc):
    """Calculates min, mean, max slope for each segment."""
    stats = []
    unique_segment_ids = np.unique(multi_seg_mask)
    unique_segment_ids = unique_segment_ids[unique_segment_ids != 0] 

    for segment_id in unique_segment_ids:

        seg_mask = (multi_seg_mask == segment_id)
        twi_values_in_segment = twi[seg_mask]
        flowacc_values_in_segment = flowacc[seg_mask]

        stats.append([
            np.min(twi_values_in_segment),
            np.mean(twi_values_in_segment),
            np.max(twi_values_in_segment),
            np.std(twi_values_in_segment),

            np.min(flowacc_values_in_segment),
            np.sum(flowacc_values_in_segment),
            np.max(flowacc_values_in_segment),
            np.std(flowacc_values_in_segment),
        ])

    return np.array(stats, dtype=np.float32)
