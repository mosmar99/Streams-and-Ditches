import os
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

################## READ IMAGES ##################

labels_dir = "../../data/data_sample/training/labels"
slopes_dir = "../../data/data_sample/training/slope"
file_names = [f for f in os.listdir(labels_dir) if f.endswith(".tif")]
slopes = []
labels = []
for file in file_names:
    file_path_slope = os.path.join(slopes_dir, file)
    file_path_label = os.path.join(labels_dir, file)
    image_slope = Image.open(file_path_slope).convert("F")
    image_label = Image.open(file_path_label).convert("L")
    image_slope_array = np.array(image_slope)
    image_label_array = np.array(image_label)
    slopes.append(image_slope_array)
    labels.append(image_label_array)

slopes = np.array(slopes)
labels = np.array(labels)

def show_image(image_array):
    # ditches=1, streams=2
    plt.imshow(image_array, cmap="gray")
    plt.title("Image")
    plt.axis("off")
    plt.show()
show_image(slopes[0])

################## SPLIT TRAIN/VAL/TEST ##################

SPLIT_TRAIN_VAL = 0.8
SPLIT_TRAIN = 0.9

def shuffle_and_split_indices(labels, split_ratio):
    shuffled_indices = np.arange(0, len(labels), 1)
    np.random.shuffle(shuffled_indices)
    split_index = round(len(labels) * split_ratio)
    return shuffled_indices[:split_index], shuffled_indices[split_index:]

train_val_indices, test_indices = shuffle_and_split_indices(labels, SPLIT_TRAIN_VAL)
train_indices, val_indices = shuffle_and_split_indices(train_val_indices, SPLIT_TRAIN)

print('Train Fraction', round(len(train_indices) / len(labels), 3))
print('Val Fraction', round(len(val_indices)/ len(labels), 3))
print('Test Fraction', round(len(test_indices)/ len(labels), 3))

train_slopes = slopes[train_indices]
train_labels = labels[train_indices]

val_slopes = slopes[val_indices]
val_labels = labels[val_indices]

test_slopes = slopes[test_indices]
test_labels = labels[test_indices]

print(slopes[0][0])

################## MODEL ##################
