import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

labels_dir = "./sample_data/training/labels"
label_files = [f for f in os.listdir(labels_dir) if f.endswith(".tif")]
mean_intensities = []
for file in label_files:
    file_path = os.path.join(labels_dir, file)
    image = Image.open(file_path).convert("L")
    image_array = np.array(image)

    mean_intensity = np.mean(image_array)
    mean_intensities.append(mean_intensity)

file_path = os.path.join(labels_dir, label_files[0])
image = Image.open(file_path).convert("L")
image_array = np.array(image)

print(image_array.shape)
print(image_array)
with open('temp.log', 'w') as log_file:
    for row in image_array:
        log_file.write(np.array2string(row))

mean_value = np.mean(mean_intensities)
std_value = np.std(mean_intensities)
print(mean_value, std_value)

plt.figure(figsize=(10, 6))
plt.hist(mean_intensities, bins=30, edgecolor='black', alpha=0.7)
plt.title(f"Distribution of Mean Intensities\nMean: {mean_value:.2f}, Std: {std_value:.2f}")
plt.xlabel("Mean Intensity")
plt.ylabel("Frequency")
plt.grid()
plt.savefig("mean_intensity_distribution.png")
plt.show()