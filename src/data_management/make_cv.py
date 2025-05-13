# Mahmut Osmanovic, Isac Paulsson, 2025 April/May

import os
import numpy as np

def write_to_file(file_list, file_name):
    with open(file_name, 'w') as f:
        for item in file_list:
            f.write(os.path.splitext(item)[0] + '\n')

def make_cv(file_names, folds_dir, k=10):
    os.makedirs(folds_dir, exist_ok=True)

    np.random.shuffle(file_names)
    num_images = len(file_names)
    images_per_fold = num_images // k

    for i in range(k):
        start_index = images_per_fold * i
        if i == k:
            test_set = file_names[start_index :]
        else:
            test_set = file_names[start_index : start_index + images_per_fold]
        train_set = [image for image in file_names if image not in test_set]

        fold_dir = os.path.join(folds_dir, f"f{i+1}")
        os.makedirs(fold_dir, exist_ok=True)

        write_to_file(test_set, os.path.join(fold_dir, "test.dat"))
        write_to_file(train_set, os.path.join(fold_dir, "train.dat"))

def make_rcv(file_names, folds_dir, k=10, r=10):
    for i in range(r):
        fold_dir_r = os.path.join(folds_dir, f"r{i+1}")
        make_cv(file_names, fold_dir_r, k=k)

if __name__ == '__main__':
    labels_dir = "./data/05m_chips/labels"
    folds_dir = "./data/05m_folds_mapio"

    file_names = [f for f in os.listdir(labels_dir) if f.endswith(".tif")]

    make_rcv(file_names, folds_dir, k=10, r=10)


