import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class Visualize:
    @staticmethod
    def visualize_k_random(k, dataset):
        """
        Visualize k random images from the dataset.

        Parameters:
        - k: Number of random images to visualize.
        - dataset: The dataset from which to sample images.
        """
        num_samples = min(k, len(dataset))  # Ensure we don't exceed the dataset size
        random_indices = random.sample(range(len(dataset)), num_samples)

        for idx in random_indices:
            image_tensor, label_tensor, file_name = dataset[idx]
            
            # Convert the image tensor to a NumPy array and scale it to [0, 255]
            image = image_tensor.numpy().squeeze()  # Remove channel dimension if present
            image = (image * 255).astype('uint8')  # Scale to [0, 255]

            # Print height and width
            height, width = image.shape
            print(f"Image {file_name}: Height = {height}, Width = {width}")

            # Display the image using OpenCV
            cv2.imshow(f"Image: {file_name}", image)
            cv2.waitKey(0)  # Wait for a key press to close the window

        cv2.destroyAllWindows()  # Close all OpenCV windows

    @staticmethod
    def plot_class_distribution(dataset):
        """
        Count occurrences of each class in the dataset and plot the distribution.

        Parameters:
        - dataset: The dataset from which to count class occurrences.
        """
        all_labels = []
        total_images = len(dataset)

        for idx, (_, label_tensor, _) in enumerate(dataset):
            label_tensor = label_tensor.squeeze(0)
            label_array = label_tensor.numpy()

            unique_classes, counts = np.unique(label_array, return_counts=True)

            for cls, count in zip(unique_classes, counts):
                all_labels.append((cls, count))

            if (idx + 1) % (total_images // 20) == 0:
                print(f"Processed {idx + 1}/{total_images} images ({(idx + 1) / total_images * 100:.2f}%)")

        class_counts = Counter(dict(all_labels))

        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        class_data = {cls: [] for cls in classes}
        for idx, (_, label_tensor, _) in enumerate(dataset):
            label_tensor = label_tensor.squeeze(0)
            label_array = label_tensor.numpy()
            unique_classes, counts = np.unique(label_array, return_counts=True)
            for cls, count in zip(unique_classes, counts):
                class_data[cls].append(count)

        plt.figure(figsize=(10, 6))

        # Create violin plot
        parts = plt.violinplot(list(class_data.values()), showmeans=True, showmedians=True)

        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        # Overlay mean
        means = []
        for i, cls in enumerate(classes):
            mean = np.mean(class_data[cls])
            means.append(mean)
            plt.scatter([i + 1], [mean], color='red', label='Mean' if i == 0 else "", zorder=3)

        # Set x-axis labels
        plt.xticks(range(1, len(classes) + 1), ['Background', 'Ditch', 'Stream'])

        plt.xlabel('Classes', fontsize=14)
        plt.ylabel('Occurrences', fontsize=14)
        plt.title('Class Distributions of Labels', fontsize=16)

        # Create a legend with mean values
        mean_text = '\n'.join([f'{label}: {round(mean)} pixels' for label, mean in zip([' - Background', ' - Ditch', ' - Stream'], means)])
        plt.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)], 
                   [f'Mean Pixel Count:\n{mean_text}'], loc='upper right')

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()