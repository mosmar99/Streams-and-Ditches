import os
from PIL import Image
from .augmentations import ImageAugmentation
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class UNetDataset(Dataset):
    def __init__(self, file_list, image_folder, label_folder, augmentations=None):
        self.file_list = file_list
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.augmentations = augmentations
        if self.augmentations:
            self.num_versions_per_file = len(self.augmentations)
        else:
            self.num_versions_per_file = 1

    def __len__(self):
        return len(self.file_list) * self.num_versions_per_file

    def __getitem__(self, idx):
        # Determine the original index and augmentation index
        original_idx = idx // self.num_versions_per_file
        augmentation_idx = idx % self.num_versions_per_file

        file_name = self.file_list[original_idx]
        
        # Use os.path.join to construct paths correctly
        image_path = self.image_folder + '/' + file_name + '.tif'
        label_path = self.label_folder + '/' + file_name + '.tif'

        image = Image.open(image_path).convert("F")
        label = Image.open(label_path).convert("F")

        # Apply augmentations if provided
        if self.augmentations is not None:
            augmented_images, augmented_labels = self.augmentations.apply_augmentations(image, label)
            return augmented_images[augmentation_idx], augmented_labels[augmentation_idx]

        # If no augmentations, return the original
        return transforms.ToTensor()(image), transforms.ToTensor()(label)

def load_data(batch_size,
              num_workers=7,
              pin_memory=True,
              train_fnames_path='./data/cv/r1_k10/r1/k1/train.dat',
              val_fnames_path=None,
              test_fnames_path='./data/cv/r1_k10/r1/k1/test.dat',
              image_folder='./data/raw/05m_chips/slope',
              label_folder='./data/raw/05m_chips/labels/',
              augmentations=None):

    if num_workers is None:
        num_workers = max(0, os.cpu_count() - 1)

    def read_file_list(file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    print("Beginning to load data...")
    train_files = read_file_list(train_fnames_path)
    test_files = read_file_list(test_fnames_path)
    
    if val_fnames_path is not None:
        val_files = read_file_list(val_fnames_path)
    else:
        val_files = None
    
    train_dataset = UNetDataset(train_files, image_folder, label_folder, augmentations=augmentations)
    test_dataset = UNetDataset(test_files, image_folder, label_folder, augmentations=augmentations)

    if val_files is not None:
        val_dataset = UNetDataset(val_files, image_folder, label_folder, augmentations=augmentations)
    else:
        val_dataset = None
    
    print(f" -- Train dataset size: {len(train_dataset)} / {len(train_dataset) + len(test_dataset)} ~ {round(len(train_dataset) / (len(train_dataset) + len(test_dataset)), 2)}")
    print(f" -- Test dataset size: {len(test_dataset)} / {len(train_dataset) + len(test_dataset)} ~ {round(len(test_dataset) / (len(train_dataset) + len(test_dataset)), 2)}")
    print(f" -- Created DataLoaders with batch size: {batch_size}")
    
    if val_dataset is not None:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader, train_dataset, test_dataset

if __name__ == "__main__":
    # Example parameters for loading data
    batch_size = 8
    augmentations = ImageAugmentation()
    train_loader, test_loader, train_dataset, test_dataset = load_data(batch_size, augmentations=augmentations)
    
    # Check if datasets are empty
    if len(train_dataset) == 0:
        print("Warning: Train dataset is empty!")
    if len(test_dataset) == 0:
        print("Warning: Test dataset is empty!")

    # Check the shape of a few images
    def check_image_shapes(dataset, num_samples=5):
        for i in range(num_samples):
            image_tensor, label_tensor, file_name = dataset[i]
            print(f"Image {i}: Shape = {image_tensor.shape}, Label Shape = {label_tensor.shape}")

    # Check shapes of the first few images in the training dataset
    check_image_shapes(train_dataset)