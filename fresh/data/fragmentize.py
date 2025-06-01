import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import math
import os

class FragmentedUNetDataset(Dataset):
    def __init__(self, base_dataset, image_size=512, patch_size_factor=0.25):
        self.base_dataset = base_dataset
        self.image_size = image_size
        self.patch_size_factor = patch_size_factor

        self.patch_size = int(image_size * patch_size_factor)
        if self.patch_size <= 0:
            raise ValueError("Patch size is too small (zero or negative).")

        self.patches_per_row = math.ceil(image_size / self.patch_size)
        self.patches_per_image = self.patches_per_row ** 2

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.base_dataset) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image

        fname = self.base_dataset.file_list[img_idx]
        img_path = os.path.join(self.base_dataset.image_folder, fname + '.tif')
        label_path = os.path.join(self.base_dataset.label_folder, fname + '.tif')

        img = Image.open(img_path).convert("F")
        label = Image.open(label_path).convert("F")

        px = (patch_idx % self.patches_per_row) * self.patch_size
        py = (patch_idx // self.patches_per_row) * self.patch_size

        img_patch = img.crop((px, py, px + self.patch_size, py + self.patch_size))
        label_patch = label.crop((px, py, px + self.patch_size, py + self.patch_size))

        return self.to_tensor(img_patch), self.to_tensor(label_patch), fname
