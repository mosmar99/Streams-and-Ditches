import math
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# costants
NUM_CLASSES = 3

# HPTs
num_epochs = 20
batch_size = 15
learning_rate = 0.001

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding='same'),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding='same'),
        nn.ReLU(inplace=True),
    )
    return conv

def crop_img(og_tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size  = og_tensor.size()[2]
    diff_size = tensor_size - target_size
    delta = diff_size // 2
    if diff_size % 2 == 0:
        return og_tensor[:, :, delta:(tensor_size-delta), delta:(tensor_size-delta)]
    return og_tensor[:, :, delta:(tensor_size-delta-1), delta:(tensor_size-delta-1)]


def add_padding(inputs):
    '''Add symmetric zero-padding to make H and W powers of 2'''
    _, _, height, width = inputs.size()

    height_correct = 0 if (height & (height - 1) == 0) and height != 0 else 2 ** math.ceil(math.log2(height)) - height
    width_correct = 0 if (width & (width - 1) == 0) and width != 0 else 2 ** math.ceil(math.log2(width)) - width

    pad_top = height_correct // 2
    pad_bottom = height_correct - pad_top
    pad_left = width_correct // 2
    pad_right = width_correct - pad_left

    padding = (pad_left, pad_right, pad_top, pad_bottom)
    padded_inputs = nn.functional.pad(inputs, padding, mode='constant', value=0)

    return padded_inputs, padding

def remove_padding(tensor, padding):
    '''Remove padding from (left, right, top, bottom)'''
    left, right, top, bottom = padding

    bottom = None if bottom == 0 else -bottom
    right = None if right == 0 else -right

    return tensor[:, :, top:bottom, left:right]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = double_conv(1, 32)
        self.down_conv2 = double_conv(32, 64)
        self.down_conv3 = double_conv(64, 128)
        self.down_conv4 = double_conv(128, 256)
        self.down_conv5 = double_conv(256, 512)
        # self.down_conv5 = double_conv(512, 1024)

        # self.up_trans_0 = nn.ConvTranspose2d(
        #     in_channels=1024,
        #     out_channels=512,
        #     kernel_size=2,
        #     stride=2)
        # self.up_conv_0 = double_conv(1024, 512)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2)
        self.up_conv_1 = double_conv(512, 256)

        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2)
        self.up_conv_2 = double_conv(256, 128)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2)
        self.up_conv_3 = double_conv(128, 64)

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=2,
            stride=2)
        self.up_conv_4 = double_conv(64, 32)

        self.out = nn.Conv2d(
            in_channels=32,
            out_channels=NUM_CLASSES,
            kernel_size=1,
        )

    def forward(self, image):
        # bs, c, h, w = image.shape
        # encoder
        x1 = self.down_conv1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv5(x8)


        # decoder
        x = self.up_trans_1(x9)
        y = crop_img(x7, x)
        x = self.up_conv_1(torch.cat([x, y], dim=1))

        x = self.up_trans_2(x)
        y = crop_img(x5, x)
        x = self.up_conv_2(torch.cat([x, y], dim=1))

        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        x = self.up_conv_3(torch.cat([x, y], dim=1))

        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        x = self.up_conv_4(torch.cat([x, y], dim=1))


        x = self.out(x)

        return x

class UNetDataset(Dataset):
    def __init__(self, file_list, image_folder, label_folder):
        self.file_list = file_list
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        image_path = os.path.join(self.image_folder, file_name) + '.tif'
        label_path = os.path.join(self.label_folder, file_name) + '.tif'

        image = Image.open(image_path).convert("F")
        label = Image.open(label_path).convert("F") 
        
        # image = tiff.imread(image_path)
        # label = tiff.imread(label_path)

        image_tensor = self.transform(image)
        label_tensor = self.transform(label)

        return image_tensor, label_tensor

def calculate_iou(outputs, labels, num_classes):
    preds = torch.argmax(outputs, dim=1)
    # flatten the predictions and labels
    preds = preds.view(-1)
    labels = labels.view(-1)
    intersection_counts = []
    union_counts = []
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)

        intersection = torch.logical_and(pred_mask, label_mask).sum().float()
        union = torch.logical_or(pred_mask, label_mask).sum().float()

        intersection_counts.append(intersection.item())
        union_counts.append(union.item())

    return intersection_counts, union_counts

if __name__ == "__main__":
    begin_time = time.time()

    def read_file_list(file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    # NAMES: Read file lists
    test_fnames_path = './data/mapio_folds/f1/test_files.dat'
    train_fnames_path = './data/mapio_folds/f1/train_files.dat'
    val_fnames_path = './data/mapio_folds/f1/val_files.dat'
    print("Beginning to load data...")
    train_files = read_file_list(train_fnames_path)
    test_files = read_file_list(test_fnames_path)
    val_files = read_file_list(val_fnames_path)

    # Create Datasets
    image_folder = './data/05m_chips/slope/'
    label_folder = './data/05m_chips/labels/'
    train_dataset = UNetDataset(train_files, image_folder, label_folder)
    test_dataset = UNetDataset(test_files, image_folder, label_folder)
    val_dataset = UNetDataset(val_files, image_folder, label_folder)
    print(f"Created datasets with {len(train_dataset)} train, {len(test_dataset)} test, {len(val_dataset)} val samples.")

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=15, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=15, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=15, pin_memory=True)
    print(f"Created DataLoaders with batch size {batch_size}")

    # Create a UNet model
    model = UNet().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train() 
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # add padding to images and labels
            images, padding = add_padding(images)
            labels, _ = add_padding(labels) 
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels.squeeze(1).long()) # Labels need to be LongTensor of shape (N, H, W)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0) # Accumulate loss weighted by batch size

            if (i+1) % 10 == 0:
                 print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item()}", flush=True)

        epoch_loss = running_loss / len(train_dataset) # Calculate average loss for the epoch
        print(f" -- Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {epoch_loss:.4f}", flush=True)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        total_intersections = [0] * NUM_CLASSES
        total_unions = [0] * NUM_CLASSES
        with torch.no_grad():
            for images, labels in val_loader:
                images, padding = add_padding(images)
                labels, _ = add_padding(labels) 
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels.squeeze(1).long())
                val_loss += loss.item() * images.size(0)

                batch_intersections, batch_unions = calculate_iou(outputs, labels.squeeze(1), NUM_CLASSES)
                for cls in range(NUM_CLASSES):
                    total_intersections[cls] += batch_intersections[cls]
                    total_unions[cls] += batch_unions[cls]

        val_loss /= len(val_dataset)
        avg_class_iou = []
        for cls in range(NUM_CLASSES):
            iou = total_intersections[cls] / total_unions[cls] if total_unions[cls] > 0 else 0.0
            avg_class_iou.append(iou)
        
        end_time = time.time()
        print(f" -- Validation Loss: {val_loss:.4f}, Average Validation IoU per class: {avg_class_iou}", flush=True)
        print(f" -- Time: {end_time - start_time:.2f} seconds\n", flush=True)

    final_time = time.time()
    print(f"Training completed in {(final_time - begin_time) / 60:.2f} minutes", flush=True)

    # Save the model checkpoint
    torch.save(model.state_dict(), 'unet_model.pth')
    print("Model saved as unet_model.pth")