import os
import tifffile
import numpy as np
from tqdm import tqdm
from skimage.filters import gaussian
from skimage.morphology import skeletonize

def expand_labels(source):
    output = np.zeros_like(source)
    for label in np.unique(source):
        if label == 0:
            continue
        target = source == label
        skel_target = skeletonize(target)
        gauss_skel = gaussian(skel_target, sigma=2)
        gauss_skel_thresh = gauss_skel > 0.10
        output[gauss_skel_thresh] = label
    return output

def main(source_path, labels_dir, target_dir):
    gt_path = os.path.join(source_path, labels_dir)

    os.makedirs(os.path.join(source_path, target_dir), exist_ok=True)

    for image in tqdm(os.listdir(gt_path)):
        source_labels = tifffile.imread(os.path.join(source_path, labels_dir, image))
        new_labels = expand_labels(source_labels)
        tifffile.imwrite(os.path.join(source_path, target_dir, image), new_labels)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Expand skeletonized labels in TIFF images.")
    parser.add_argument('source_path', type=str, help='Root directory containing label subfolder')
    parser.add_argument('labels_dir', type=str, help='Subdirectory name containing input label TIFFs')
    parser.add_argument('target_dir', type=str, help='Subdirectory name to save output TIFFs')

    args = parser.parse_args()
    main(args.source_path, args.labels_dir, args.target_dir)

# python expand_labels.py ./data/05m_chips labels new_labels
