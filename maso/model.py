# 1) Design Model (input size, output size, forward pass)
# 2) Define Loss Function and Optimizer
# 3) Training Loop
#   - forward pass: compute prediction and loss
#   - backward pass: compute gradients
#   - update weights

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# costants
NUM_CLASSES = 3

# HPTs
num_epochs = 100
batch_size = 16
learning_rate = 0.001

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
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

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = double_conv(1, 64)
        self.down_conv2 = double_conv(64, 128)
        self.down_conv3 = double_conv(128, 256)
        self.down_conv4 = double_conv(256, 512)
        self.down_conv5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024, 
            out_channels=512, 
            kernel_size=2, 
            stride=2)
        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512, 
            out_channels=256, 
            kernel_size=2, 
            stride=2)
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256, 
            out_channels=128, 
            kernel_size=2, 
            stride=2)
        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128, 
            out_channels=64, 
            kernel_size=2, 
            stride=2)
        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=NUM_CLASSES,
            kernel_size=1,
        )

    def forward(self, image):
        # bs, c, h, w = image.shape
        # encoder
        x1 = self.down_conv1(image) #
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv2(x2) #
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv3(x4) #
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv4(x6) #
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

        print(f"{x}")

        return x


if __name__ == "__main__":
    image_path_slope = './data/05m_chips/slope/18E023_68900_6125_25_0001.tif'
    image_slope = Image.open(image_path_slope).convert("F")
    image_slope = transforms.ToTensor()(image_slope).unsqueeze(0).to(device)
    print(f"Image shape: {image_slope.shape}")

    model = UNet().to(device)
    output = model(image_slope)
    # print(f"Output shape: {output.shape}")
