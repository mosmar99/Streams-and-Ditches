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
import argparse
import unet
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import center_of_mass
from skimage.segmentation import watershed
from skimage.morphology import remove_small_holes, skeletonize
from skimage import feature

NUM_CLASSES = 3

num_epochs = 100
batch_size = 8
learning_rate = 0.001
device = "cuda"

if __name__ == "__main__": 
    begin_time = time.time()

    def read_file_list(file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    test_fnames_path = './data/mapio_folds/f1/test_files.dat'
    train_fnames_path = './data/mapio_folds/f1/train_files.dat'
    val_fnames_path = './data/mapio_folds/f1/val_files.dat'
    print("Beginning to load data...")
    train_files = read_file_list(train_fnames_path)
    test_files = read_file_list(test_fnames_path)
    val_files = read_file_list(val_fnames_path)

    image_folder = './data/05m_chips/slope/'
    label_folder = './data/05m_chips/labels/'
    train_dataset = unet.UNetDataset(train_files, image_folder, label_folder)
    test_dataset = unet.UNetDataset(test_files, image_folder, label_folder)
    val_dataset = unet.UNetDataset(val_files, image_folder, label_folder)
    print(f"Created datasets with {len(train_dataset)} train, {len(test_dataset)} test, {len(val_dataset)} val samples.")

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=15, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=15, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=15, pin_memory=True)
    print(f"Created DataLoaders with batch size {batch_size}")
    # "./logs/m1/20250513_161035"
    best_model_path = os.path.join("./logs/m1/20250515_134712", 'unet_model_ckpt.pth')
    if os.path.exists(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        test_model = unet.UNet().to(device)
        test_model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_model.eval()

        optimizer = torch.optim.AdamW(test_model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, padding = unet.add_padding(images)
            labels_padded, _ = unet.add_padding(labels)
            images = images.to(device)
            squeezed_labels_padded = labels_padded.squeeze(1).long().to(device)

            outputs = torch.softmax(test_model(images), dim=1)
            # loss = criterion(outputs, squeezed_labels_padded)
            print(outputs[0].shape)
            class_outputs = torch.argmax(outputs, 1)
            print(outputs[0].shape)
            print(labels[0].shape)

            # fig, ax = plt.subplots(2,2, figsize=(15, 10))
            # ax[0][0].imshow(outputs[0,1].cpu().numpy(), cmap="Greys")
            # ax[0][0].set_title("ditch_probabilities")
            # ax[0][0].axis("off")
            # ax[0][1].imshow(outputs[0,2].cpu().numpy(), cmap="Greys")
            # ax[0][1].set_title("stream_probabilities")
            # ax[0][1].axis("off")
            # ax[1][0].imshow(class_outputs[0].cpu().numpy(), cmap="Greys")
            # ax[1][0].set_title("Predictions")
            # ax[1][0].axis("off")
            # ax[1][1].imshow(labels[0].cpu().squeeze(0).numpy(), cmap="Greys")
            # ax[1][1].set_title("Targets")
            # ax[1][1].axis("off")
            # plt.show()

            bg = outputs[0,1].cpu().numpy() + outputs[0,2].cpu().numpy()
            bg_thresh = bg > 0.15
            skel_bg = skeletonize(bg_thresh)
            fig, ax = plt.subplots(2,2, figsize=(15, 10))
            ax[0][0].imshow(bg, cmap="Greys")
            ax[0][0].set_title("background_probabilities")
            ax[0][0].axis("off")
            ax[0][1].imshow(bg_thresh, cmap="Greys")
            ax[0][1].set_title("bakground threshold")
            ax[0][1].axis("off")
            ax[1][0].imshow(skel_bg, cmap="Greys")
            ax[1][0].set_title("Predictions")
            ax[1][0].axis("off")
            ax[1][1].imshow(labels[0].cpu().squeeze(0).numpy(), cmap="Greys")
            ax[1][1].set_title("Targets")
            ax[1][1].axis("off")
            plt.show()

            # test_loss += loss.item() * images.size(0)