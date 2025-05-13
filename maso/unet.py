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

parser = argparse.ArgumentParser(description='UNet Training')
parser.add_argument('--logdir', type=str, default='logs', help='Directory to save logs')
args = parser.parse_args()
logdir = args.logdir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLASSES = 3

num_epochs = 100
batch_size = 8
learning_rate = 0.001

print(f"num_epochs: {num_epochs}, batch_size: {batch_size}, learning_rate: {learning_rate}")

def double_conv(in_c, out_c, dropout_prob=0.1): 
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
    target_size = target_tensor.size()[2]
    tensor_size  = og_tensor.size()[2]
    diff_size = tensor_size - target_size
    delta = diff_size // 2
    if diff_size % 2 == 0:
        return og_tensor[:, :, delta:(tensor_size-delta), delta:(tensor_size-delta)]
    return og_tensor[:, :, delta:(tensor_size-delta-1), delta:(tensor_size-delta-1)]


def add_padding(inputs):
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
        x1 = self.down_conv1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv5(x8)

        x = self.up_trans_1(x9)
        y = crop_img(x7, x)
        x = self.up_conv_1(torch.cat([x, y], dim=1))

        x = self.up_trans_2(x) # Corrected: input from previous upsampling block
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

        image_tensor = self.transform(image)
        label_tensor = self.transform(label)

        return image_tensor, label_tensor

def calculate_iou(outputs_logits, labels, num_classes):
    preds = torch.argmax(outputs_logits, dim=1)
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

def calculate_recall(outputs_logits, labels, num_classes):
    preds = torch.argmax(outputs_logits, dim=1)
    preds = preds.view(-1)
    labels = labels.view(-1)

    batch_true_positives = []
    batch_actual_positives = []

    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)

        tp = torch.logical_and(pred_mask, label_mask).sum().item()
        batch_true_positives.append(tp)

        actual_positives = label_mask.sum().item()
        batch_actual_positives.append(actual_positives)

    return batch_true_positives, batch_actual_positives

def calculate_f1_components(outputs_logits, labels, num_classes):
    preds = torch.argmax(outputs_logits, dim=1).view(-1)
    labels = labels.view(-1)
    
    batch_tp = []
    batch_fp = []
    batch_fn = []

    for cls in range(num_classes):
        pred_mask_cls = (preds == cls)
        label_mask_cls = (labels == cls)
        
        tp = torch.logical_and(pred_mask_cls, label_mask_cls).sum().item()
        fp = torch.logical_and(pred_mask_cls, ~label_mask_cls).sum().item()
        fn = torch.logical_and(~pred_mask_cls, label_mask_cls).sum().item()
        
        batch_tp.append(tp)
        batch_fp.append(fp)
        batch_fn.append(fn)
        
    return batch_tp, batch_fp, batch_fn

def calculate_mcc_components(outputs_logits, labels, num_classes):
    preds = torch.argmax(outputs_logits, dim=1).view(-1)
    labels_flat = labels.view(-1)
    
    batch_tp = []
    batch_tn = []
    batch_fp = []
    batch_fn = []

    for cls in range(num_classes):
        pred_is_cls = (preds == cls)
        label_is_cls = (labels_flat == cls)
        
        pred_is_not_cls = ~pred_is_cls
        label_is_not_cls = ~label_is_cls
        
        tp = torch.logical_and(pred_is_cls, label_is_cls).sum().item()
        tn = torch.logical_and(pred_is_not_cls, label_is_not_cls).sum().item()
        fp = torch.logical_and(pred_is_cls, label_is_not_cls).sum().item()
        fn = torch.logical_and(pred_is_not_cls, label_is_cls).sum().item()
        
        batch_tp.append(tp)
        batch_tn.append(tn)
        batch_fp.append(fp)
        batch_fn.append(fn)
        
    return batch_tp, batch_tn, batch_fp, batch_fn


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
    train_dataset = UNetDataset(train_files, image_folder, label_folder)
    test_dataset = UNetDataset(test_files, image_folder, label_folder)
    val_dataset = UNetDataset(val_files, image_folder, label_folder)
    print(f"Created datasets with {len(train_dataset)} train, {len(test_dataset)} test, {len(val_dataset)} val samples.")

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=15, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=15, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=15, pin_memory=True)
    print(f"Created DataLoaders with batch size {batch_size}")

    model = UNet().to(device)

    weights = torch.tensor([1.0, 10.0, 100.0])
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, padding = add_padding(images)
            labels_padded, _ = add_padding(labels)
            images = images.to(device)
            squeezed_labels_padded = labels_padded.squeeze(1).long().to(device)

            outputs = model(images)
            loss = criterion(outputs, squeezed_labels_padded)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item()}", flush=True)

        epoch_loss = running_loss / len(train_dataset)
        print(f" -- Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {epoch_loss:.4f}", flush=True)

        model.eval()
        val_loss = 0.0
        total_intersections = [0] * NUM_CLASSES
        total_unions = [0] * NUM_CLASSES
        total_true_positives_recall = [0] * NUM_CLASSES
        total_actual_positives_in_label = [0] * NUM_CLASSES
        
        total_tp_f1 = [0] * NUM_CLASSES
        total_fp_f1 = [0] * NUM_CLASSES
        total_fn_f1 = [0] * NUM_CLASSES

        total_tp_mcc = [0] * NUM_CLASSES
        total_tn_mcc = [0] * NUM_CLASSES
        total_fp_mcc = [0] * NUM_CLASSES
        total_fn_mcc = [0] * NUM_CLASSES


        with torch.no_grad():
            for images, labels in val_loader:
                # setup
                images, padding = add_padding(images)
                labels_padded, _ = add_padding(labels)
                images = images.to(device)
                squeezed_labels_padded = labels_padded.squeeze(1).long().to(device)

                # loss
                outputs = model(images)
                loss = criterion(outputs, squeezed_labels_padded)
                val_loss += loss.item() * images.size(0)

                # IoU statistics
                batch_intersections, batch_unions = calculate_iou(outputs, squeezed_labels_padded, NUM_CLASSES)
                for cls in range(NUM_CLASSES):
                    total_intersections[cls] += batch_intersections[cls]
                    total_unions[cls] += batch_unions[cls]

                # Recall statistics
                batch_tp_recall, batch_ap_recall = calculate_recall(outputs, squeezed_labels_padded, NUM_CLASSES)
                for cls in range(NUM_CLASSES):
                    total_true_positives_recall[cls] += batch_tp_recall[cls]
                    total_actual_positives_in_label[cls] += batch_ap_recall[cls]
                
                # F1 components
                batch_tp_f1_val, batch_fp_f1_val, batch_fn_f1_val = calculate_f1_components(outputs, squeezed_labels_padded, NUM_CLASSES)
                for cls in range(NUM_CLASSES):
                    total_tp_f1[cls] += batch_tp_f1_val[cls]
                    total_fp_f1[cls] += batch_fp_f1_val[cls]
                    total_fn_f1[cls] += batch_fn_f1_val[cls]

                # MCC components
                batch_tp_mcc_val, batch_tn_mcc_val, batch_fp_mcc_val, batch_fn_mcc_val = calculate_mcc_components(outputs, squeezed_labels_padded, NUM_CLASSES)
                for cls in range(NUM_CLASSES):
                    total_tp_mcc[cls] += batch_tp_mcc_val[cls]
                    total_tn_mcc[cls] += batch_tn_mcc_val[cls]
                    total_fp_mcc[cls] += batch_fp_mcc_val[cls]
                    total_fn_mcc[cls] += batch_fn_mcc_val[cls]


        val_loss /= len(val_dataset)
        avg_class_iou = []
        avg_class_recall = []
        avg_class_f1 = []
        avg_class_mcc = []

        for cls in range(NUM_CLASSES):
            # iou
            iou = total_intersections[cls] / total_unions[cls] if total_unions[cls] > 0 else 0.0
            avg_class_iou.append(iou)

            # recall
            recall_val = total_true_positives_recall[cls] / total_actual_positives_in_label[cls] if total_actual_positives_in_label[cls] > 0 else 0.0
            avg_class_recall.append(recall_val)
            
            # F1-score
            tp_f1, fp_f1, fn_f1 = total_tp_f1[cls], total_fp_f1[cls], total_fn_f1[cls]
            precision_f1 = tp_f1 / (tp_f1 + fp_f1) if (tp_f1 + fp_f1) > 0 else 0.0
            recall_f1 = tp_f1 / (tp_f1 + fn_f1) if (tp_f1 + fn_f1) > 0 else 0.0
            f1_score = 2 * (precision_f1 * recall_f1) / (precision_f1 + recall_f1) if (precision_f1 + recall_f1) > 0 else 0.0
            avg_class_f1.append(f1_score)

            # MCC calculation
            tp, tn, fp, fn = total_tp_mcc[cls], total_tn_mcc[cls], total_fp_mcc[cls], total_fn_mcc[cls]
            numerator = (tp * tn) - (fp * fn)
            denominator_val = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            mcc = 0.0 if denominator_val == 0 else numerator / math.sqrt(denominator_val)
            avg_class_mcc.append(mcc)


        if not os.path.exists(logdir):
            os.makedirs(logdir)
            
        log_header = 'epoch,val_loss'
        log_values = f"{epoch + 1},{val_loss:.4f}"

        # Collect metrics for each class and format them 
        recalls = [f'{avg_class_recall[i]:.4f}' for i in range(NUM_CLASSES)]
        f1_scores = [f'{avg_class_f1[i]:.4f}' for i in range(NUM_CLASSES)]
        mccs = [f'{avg_class_mcc[i]:.4f}' for i in range(NUM_CLASSES)]
        ious = [f'{avg_class_iou[i]:.4f}' for i in range(NUM_CLASSES)]

        # Add recall columns
        log_header += ''.join([f',recall_class{i}' for i in range(NUM_CLASSES)])
        log_values += ''.join([f',{recalls[i]}' for i in range(NUM_CLASSES)])

        # Add f1 columns
        log_header += ''.join([f',f1_class{i}' for i in range(NUM_CLASSES)])
        log_values += ''.join([f',{f1_scores[i]}' for i in range(NUM_CLASSES)])

        # Add mcc columns
        log_header += ''.join([f',mcc_class{i}' for i in range(NUM_CLASSES)])
        log_values += ''.join([f',{mccs[i]}' for i in range(NUM_CLASSES)])

        # Add iou columns
        log_header += ''.join([f',iou_class{i}' for i in range(NUM_CLASSES)])
        log_values += ''.join([f',{ious[i]}' for i in range(NUM_CLASSES)])
                
        with open(f'{logdir}/training.log', 'a') as log_file:
            if epoch == 0:
                log_file.write(log_header + '\n')
            log_file.write(log_values + '\n')
            
        end_time = time.time()
        print(f" -- Validation Loss: {val_loss:.4f}", flush=True)
        print(f" -- Average Validation IoU per class:     {avg_class_iou}", flush=True)
        print(f" -- Average Validation Recall per class:  {avg_class_recall}", flush=True)
        print(f" -- Average Validation F1 per class:      {avg_class_f1}", flush=True)
        print(f" -- Average Validation MCC per class:     {avg_class_mcc}", flush=True)
        print(f" -- Time: {end_time - start_time:.2f} seconds\n", flush=True)

    final_time = time.time()
    print(f"Training completed in {(final_time - begin_time) / 60:.2f} minutes", flush=True)

    torch.save(model.state_dict(), 'unet_model.pth')
    print("Model saved as unet_model.pth")