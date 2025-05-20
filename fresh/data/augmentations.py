import torchvision.transforms as transforms
from PIL import Image

class ImageAugmentation:
    def __init__(self):
        self.augmentations = [
            transforms.RandomHorizontalFlip(p=1.0),  # Always apply
            transforms.RandomVerticalFlip(p=1.0),    # Always apply
            transforms.Lambda(lambda x: x.rotate(90)),
            transforms.Lambda(lambda x: x.rotate(180)),
            transforms.Lambda(lambda x: x.rotate(270))
        ]
        self.to_tensor = transforms.ToTensor()

    def apply_augmentations(self, image, label):
        # Store the original image and label
        augmented_images = []
        augmented_labels = []

        # Original image and label
        augmented_images.append(self.to_tensor(image))
        augmented_labels.append(self.to_tensor(label))

        # Apply each augmentation
        for augmentation in self.augmentations:
            augmented_image = augmentation(image)
            augmented_label = augmentation(label)

            augmented_images.append(self.to_tensor(augmented_image))
            augmented_labels.append(self.to_tensor(augmented_label))

        return augmented_images, augmented_labels

    def __len__(self):
        return len(self.augmentations) + 1
