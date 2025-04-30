# Mahmut Osmanovic, Isac Paulsson, 2025 April/May

import os
import numpy as np

################## READ FILE NAMES ##################

labels_dir = "./data/05m_chips/labels"
file_names = [f for f in os.listdir(labels_dir) if f.endswith(".tif")]

################## SPLIT TRAIN/VAL/TEST ##################

SPLIT_TRAIN_VAL = 0.8
SPLIT_TRAIN = 0.9

def shuffle_and_split_indices(arr, split_ratio):
    shuffled_indices = np.arange(0, len(arr), 1)
    np.random.shuffle(shuffled_indices)
    split_index = round(len(arr) * split_ratio)
    return shuffled_indices[:split_index], shuffled_indices[split_index:]

def get_train_val_test_files(file_names, split_train_val=0.8, split_train=0.9):
    train_val_indices, test_indices = shuffle_and_split_indices(file_names, SPLIT_TRAIN_VAL)
    train_indices, val_indices = shuffle_and_split_indices(train_val_indices, SPLIT_TRAIN)
    train_files = [file_names[i] for i in train_indices]
    val_files = [file_names[i] for i in val_indices]
    test_files = [file_names[i] for i in test_indices]
    return train_files, val_files, test_files

def write_to_file(file_list, file_name):
    with open(file_name, 'w') as f:
        for item in file_list:
            f.write(os.path.splitext(item)[0] + '\n')

def write_train_val_test_files(path, train_files, val_files, test_files):
    os.makedirs(path, exist_ok=True)
    write_to_file(train_files, f'{path}/train_files.dat')
    write_to_file(val_files, f'{path}/val_files.dat')
    write_to_file(test_files, f'{path}/test_files.dat')

train_files, val_files, test_files = get_train_val_test_files(file_names, SPLIT_TRAIN_VAL, SPLIT_TRAIN)

path = './data/mapio_folds/f1'
write_train_val_test_files(path, train_files, val_files, test_files)


