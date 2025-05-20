# Mahmut Osmanovic, Isac Paulsson, 2025 April/May

import os
import numpy as np

################## READ FILE NAMES ##################

# input and output directories
output_dir = "./data/cv"
input_dir = "./data/raw/05m_chips/labels"
file_names = [f for f in os.listdir(input_dir) if f.endswith(".tif")] # change to file extension you want

# setup
R = 2 # number of repetitions
k = 5 # number of folds

################## SPLIT TRAIN/VAL/TEST ##################

total_files = len(file_names)
files_per_test = total_files // k

print("Folder Location: ", output_dir)
print(f"R={R} repitions of k={k} folds.\n")
for r in range(1, R + 1):
    indices = np.arange(total_files)
    np.random.shuffle(indices)

    for i in range(k):
        test_start = i * files_per_test
        test_end = (i + 1) * files_per_test

        if i == k - 1:
            test_indices = indices[test_start:]
        else:
            test_indices = indices[test_start:test_end]

        train_indices = np.concatenate((indices[:test_start], indices[test_end:]))

        train_files = [file_names[j] for j in train_indices]
        test_files = [file_names[j] for j in test_indices]

        folder = f"{output_dir}/r{R}_k{k}/r{r}/k{i+1}"
        os.makedirs(folder, exist_ok=True)
        train_file_path = os.path.join(folder, f"train.dat")
        test_file_path = os.path.join(folder, f"test.dat")

        with open(train_file_path, "w") as f:
            for file in train_files:
                f.write(f"{os.path.splitext(file)[0]}\n")
        with open(test_file_path, "w") as f:
            for file in test_files:
                f.write(f"{os.path.splitext(file)[0]}\n")
        print(f"-- Saved train and test files for fold ({r}, {i+1}).")
    print()

