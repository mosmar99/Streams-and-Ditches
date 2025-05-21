import os
import torch
import argparse
import torch.optim as optim
import data.loading as get_raw_data
from scripts.train import train_unet
from scripts.test import test_unet
# from utils.visualization import Visualize
from data.augmentations import ImageAugmentation
from models.unet import UNet, TverskyLoss, device as model_device

def read_logdir():
    parser = argparse.ArgumentParser(description='UNet Training')
    parser.add_argument('--logdir', type=str, default='checkpoints/my_unet_run', help='Directory to save checkpoints')
    args = parser.parse_args()
    return args.logdir

def main():
    # parameters
    k = 10
    IMG_CHANNELS = 1
    NUM_CLASSES = 3
    batch_size_unet = 8
    num_epochs_unet = 1
    learning_rate_unet = 0.0008
    unet_model_filename = 'best_unet.pth'

    # # get data
    # augmentations = ImageAugmentation()  
    train_loader_unet, test_loader_unet, train_dataset_unet, test_dataset_unet = get_raw_data.load_data(batch_size_unet, augmentations=None)

    # # instantiate the UNet model
    unet_model = UNet(in_channels=IMG_CHANNELS, num_classes=NUM_CLASSES).to(model_device)

    # # define the loss function and optimizer
    tversky_alpha = 0.3 
    tversky_beta = 0.7
    criterion_unet = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, num_classes=NUM_CLASSES).to(model_device)
    optimizer_unet = optim.Adam(unet_model.parameters(), lr=learning_rate_unet)

    # # train the UNet model
    base_logdir = read_logdir()
    print("\nStarting UNet Training...", flush=True)
    trained_unet_model = train_unet(
        model=unet_model,
        train_loader=train_loader_unet,
        criterion=criterion_unet,
        optimizer=optimizer_unet,
        num_epochs=num_epochs_unet,
        device=model_device,
        logdir=base_logdir,
        model_name=unet_model_filename,
    )
    print("UNet training process completed.", flush=True)

    # print("\nStarting UNet Testing...", flush=True)
    unet_checkpoints_logdir = os.path.join(base_logdir, 'checkpoints')
    best_model_path = unet_checkpoints_logdir

    model_for_testing = UNet(in_channels=IMG_CHANNELS, num_classes=NUM_CLASSES)
    criterion_unet_test = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, num_classes=NUM_CLASSES).to(model_device)

    unet_test_results_logdir = os.path.join(base_logdir, 'test_results')
    best_model_path = os.path.join(unet_checkpoints_logdir, unet_model_filename)
    test_metrics = test_unet(
        model=model_for_testing,
        test_loader=test_loader_unet,
        criterion=criterion_unet_test,
        device=model_device,
        num_classes=NUM_CLASSES,
        checkpoint_path=best_model_path,
        logdir=base_logdir,
        output_dir=unet_test_results_logdir,
    )
    print("UNet testing process completed.", flush=True)
    print("Test Metrics:", test_metrics, flush=True)

    # Use this augmentation pipeline in your UNetDataset
    # augmentations = ImageAugmentation()  
    # train_loader_unet, test_loader_unet, train_dataset_unet, test_dataset_unet = get_raw_data.load_data(batch_size_unet, augmentations=augmentations)
    # Visualize.visualize_k_random(k, train_dataset_unet)

    # train_loader_unet, test_loader_unet, train_dataset_unet, test_dataset_unet = get_raw_data.load_data(batch_size_unet, augmentations=None)
    # Visualize.plot_class_distribution(train_dataset_unet + test_dataset_unet)

if __name__ == "__main__":
    main()