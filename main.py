from src.models.unet import UNet
from src.data_management.data_handling import MultibandDataset
import torch

in_channels = 1
classes = [0, 1, 2]
device = 'cuda'

model = UNet(in_channels, classes, device=device)

train_set = MultibandDataset(
    img_paths = ['./data/05m_chips/slope'],
    classes = classes,
    selected = './data/mapio_folds/f1/train_files.dat',
    gt_path= './data/05m_chips/labels',
)

val_set = MultibandDataset(
    img_paths = ['./data/05m_chips/slope'],
    classes = classes,
    selected = './data/mapio_folds/f1/val_files.dat',
    gt_path= './data/05m_chips/labels',
)

test_set = MultibandDataset(
    img_paths = ['./data/05m_chips/slope'],
    classes = classes,
    selected = './data/mapio_folds/f1/test_files.dat',
    gt_path= './data/05m_chips/labels',
)

seed = 42
batch_size = 4

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

