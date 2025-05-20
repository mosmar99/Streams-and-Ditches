# get raw data
import data.loading as get_raw_data
from data.augmentations import ImageAugmentation

def main():
    # parameters
    batch_size_unet = 8

    # Use this augmentation pipeline in your UNetDataset
    augmentations = ImageAugmentation()  
    train_loader_unet, test_loader_unet, train_dataset_unet, test_dataset_unet = get_raw_data.load_data(batch_size_unet, augmentations=augmentations)

    


if __name__ == "__main__":
    main()