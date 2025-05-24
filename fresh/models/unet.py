# models/unet_model.py (or your filename)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def double_conv(in_c, out_c, dropout_prob=0.05):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding='same', bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding='same', bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_prob)
    )
    return conv

def crop_img(og_tensor, target_tensor):
    og_size_h, og_size_w = og_tensor.size()[2], og_tensor.size()[3]
    target_size_h, target_size_w = target_tensor.size()[2], target_tensor.size()[3]
    diff_h = og_size_h - target_size_h
    diff_w = og_size_w - target_size_w
    delta_h = diff_h // 2
    crop_h_start = delta_h
    crop_h_end = og_size_h - (diff_h - delta_h)
    delta_w = diff_w // 2
    crop_w_start = delta_w
    crop_w_end = og_size_w - (diff_w - delta_w)
    return og_tensor[:, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end]

def add_padding(inputs):
    _, _, height, width = inputs.size()
    is_height_pow2 = (height & (height - 1) == 0) and height != 0
    is_width_pow2 = (width & (width - 1) == 0) and width != 0
    height_correct = 0 if is_height_pow2 else 2 ** math.ceil(math.log2(height)) - height
    width_correct = 0 if is_width_pow2 else 2 ** math.ceil(math.log2(width)) - width
    pad_top = height_correct // 2
    pad_bottom = height_correct - pad_top
    pad_left = width_correct // 2
    pad_right = width_correct - pad_left
    padding_tuple = (pad_left, pad_right, pad_top, pad_bottom)
    padded_inputs = F.pad(inputs, padding_tuple, mode='constant', value=0)
    return padded_inputs, padding_tuple

def remove_padding(tensor, padding_info):
    pad_left, pad_right, pad_top, pad_bottom = padding_info
    _, _, H_padded, W_padded = tensor.shape
    h_start = pad_top
    h_end = H_padded - pad_bottom
    w_start = pad_left
    w_end = W_padded - pad_right
    h_end = max(h_start, h_end)
    w_end = max(w_start, w_end)
    return tensor[:, :, h_start:h_end, w_start:w_end]

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6, num_classes=3):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        if self.num_classes is None:
             raise ValueError("num_classes in TverskyLoss must be set during initialization.")
        
        # Ensure targets are [N, H, W]
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        
        targets_long = targets.long()

        probs = torch.softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(probs)
        
        index_tensor = targets_long.unsqueeze(1)
        targets_one_hot.scatter_(1, index_tensor, 1.0)

        loss = 0.0
        for c in range(self.num_classes):
            probs_c = probs[:, c, :, :]
            targets_c = targets_one_hot[:, c, :, :]
            tp = (probs_c * targets_c).sum()
            fp = (probs_c * (1 - targets_c)).sum()
            fn = ((1 - probs_c) * targets_c).sum()
            tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            class_loss = 1.0 - tversky_index
            loss += class_loss
        return loss / self.num_classes

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, dropout=0.05):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = double_conv(in_channels, 32, dropout_prob=dropout)
        self.down_conv2 = double_conv(32, 64, dropout_prob=dropout)
        self.down_conv3 = double_conv(64, 128, dropout_prob=dropout)
        self.down_conv4 = double_conv(128, 256, dropout_prob=dropout)
        self.down_conv5 = double_conv(256, 512, dropout_prob=dropout)

        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=4)

        self.up_trans_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_1 = double_conv(512, 256)
        self.up_trans_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_2 = double_conv(256, 128)
        self.up_trans_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv_3 = double_conv(128, 64)
        self.up_trans_4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv_4 = double_conv(64, 32)
        self.out = nn.Conv2d(32, self.num_classes, kernel_size=1)

    def forward(self, image):
        x1 = self.down_conv1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv5(x8)

        B, C, H, W = x9.shape
        x9_flat = x9.view(B, C, H*W).permute(2, 0, 1)  # [H*W, B, C]
        attn_out, _ = self.attn(x9_flat, x9_flat, x9_flat)
        x9 = attn_out.permute(1, 2, 0).view(B, C, H, W)

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

    def predict_logits(self, images_unpadded):
        original_device = images_unpadded.device
        images_padded, padding_info = add_padding(images_unpadded.to(device))
        self.eval()
        with torch.no_grad():
            outputs_padded = self(images_padded)
        outputs_unpadded = remove_padding(outputs_padded, padding_info)
        return outputs_unpadded.to(original_device)
    
    def predict_softmax(self, images_unpadded):
        logits = self.predict_logits(images_unpadded)
        return F.softmax(logits, dim=1)