import os
import torch
import logging
import warnings
import numpy as np
from src.models.unet import UNet
from torchvision import transforms
import src.data_management.data_handling as dh
from src.data_management.data_handling import MultibandDataset

def enable_logging(log_path, file_name):
    '''Setup logging

    Parameters
    ----------
    log_path : Path to store log file
    file_name : Log file name

    '''
    # setup logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        logging.basicConfig(
                    filename=os.path.join(log_path, file_name),
                    filemode='w', level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s %(name)s - %(levelname)s - %(message)s')

def main(log_dir, epochs):
    seed = 42
    batch_size = 4
    in_channels = 1
    classes = "0,1,2"
    device = 'cuda'

    enable_logging(log_dir, 'train.log')
    rng = np.random.default_rng(seed)

    transform_train=transforms.Compose([
                                    dh.RandomFlip(rng),
                                    dh.RandomRotate(rng),
                                    dh.ToTensor(),
                                    dh.ToOnehotGaussianBlur(7,
                                                            classes,
                                                            False)])

    train_set = MultibandDataset(
        img_paths = ['./data/05m_chips/slope'],
        classes = classes,
        selected = './data/mapio_folds/f1/train_files.dat',
        gt_path= './data/05m_chips/labels',
        transform=transform_train,
    )

    transform_val=transforms.Compose([
                                    dh.ToTensor(),
                                    dh.ToOnehotGaussianBlur(7,
                                                            classes,
                                                            False)])

    val_set = MultibandDataset(
        img_paths = ['./data/05m_chips/slope'],
        classes = classes,
        selected = './data/mapio_folds/f1/val_files.dat',
        gt_path= './data/05m_chips/labels',
        transform=transform_val,
    )

    test_set = MultibandDataset(
        img_paths = ['./data/05m_chips/slope'],
        classes = classes,
        selected = './data/mapio_folds/f1/test_files.dat',
        gt_path= './data/05m_chips/labels',
    )

    train_it = torch.utils.data.DataLoader(
                                    train_set, shuffle=True,
                                    batch_size=batch_size, num_workers=0,
                                    generator=torch.Generator('cuda')
                                                    .manual_seed(seed))
    val_it = torch.utils.data.DataLoader(
                                    val_set, shuffle=True,
                                    batch_size=batch_size, num_workers=0,
                                    generator=torch.Generator('cuda')
                                                    .manual_seed(seed))
    test_it = torch.utils.data.DataLoader(
                                    test_set, shuffle=True,
                                    batch_size=batch_size, num_workers=0,
                                    generator=torch.Generator('cuda')
                                                    .manual_seed(seed))


    model = UNet(in_channels, dh.MultibandDataset.parse_classes(classes))

    logging.info('Start training')
    model.fit(train_it, val_it, epochs, log_dir,
                train_set.infer_weights('mfb'))
    logging.info('End training')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train a UNet model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('log_dir', help='Folder to write log data to')
    parser.add_argument('--epochs', help='Number of epochs to train', type=int,
                        default=10)

    args = vars(parser.parse_args())
    main(**args)
