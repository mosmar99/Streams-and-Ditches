import random
import cv2

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
