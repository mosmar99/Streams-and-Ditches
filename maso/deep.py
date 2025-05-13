import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
# Updated import for DeepLabV3+ models and weights
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.transforms.functional import resize, InterpolationMode # For resizing labels correctly

from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import numpy as np # Added for potential alpha calculation

# argparse
parser = argparse.ArgumentParser(description='DeepLabV3+ with Focal Loss Training') # Updated description
parser.add_argument('--logdir', type=str, default='logs_deeplab_focal', help='Directory to save logs') # Changed default logdir
parser.add_argument('--alpha', type=float, default=None, nargs='+', help='Alpha weights for Focal Loss per class (e.g., 0.25 0.5 0.75). If None, calculated inversely proportional to class frequency.')
parser.add_argument('--gamma', type=float, default=2.0, help='Gamma focusing parameter for Focal Loss')
parser.add_argument('--resize_size', type=int, default=513, help='Size to resize images and labels for DeepLab input') # Common size for DeepLab

args = parser.parse_args()
logdir = args.logdir
resize_size = (args.resize_size, args.resize_size)

# Create log directory if it doesn't exist
os.makedirs(logdir, exist_ok=True)

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# costants
NUM_CLASSES = 3

# HPTs - Adjusted for DeepLabV3+
num_epochs = 100
batch_size = 6  # Might need adjustment based on GPU memory and resize_size
learning_rate = 1e-4 # Often slightly lower for fine-tuning pre-trained models

print(f"num_epochs: {num_epochs}, batch_size: {batch_size}, learning_rate: {learning_rate}")
print(f"Focal Loss params: gamma={args.gamma}, alpha={args.alpha if args.alpha else 'Calculated from data'}")
print(f"Resize Size: {resize_size}")

# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for multi-class segmentation.
        Args:
            alpha (Tensor, optional): Weights for each class (C,). If None, defaults to uniform weights.
            gamma (float, optional): Focusing parameter. Defaults to 2.0.
            reduction (str, optional): Specifies the reduction to apply to the output:
                                       'none' | 'mean' | 'sum'. 'mean': outputs mean of losses,
                                       'sum': outputs sum of losses, 'none': outputs loss per element.
                                       Defaults to 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # Register alpha as a buffer to move it to the correct device automatically
        self.register_buffer('alpha', alpha)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass.
        Args:
            inputs (Tensor): Logits of shape (N, C, H, W).
            targets (Tensor): Ground truth labels of shape (N, H, W) with values in [0, C-1].
        """
        # Calculate Cross Entropy Loss without reduction
        # Use log_softmax for numerical stability
        log_prob = F.log_softmax(inputs, dim=1)
        # Use nll_loss with ignore_index=-1 if necessary, assumes valid targets here
        ce_loss = F.nll_loss(log_prob, targets, reduction='none')

        # Calculate p_t (probability of the true class)
        # Gather the probabilities corresponding to the true classes
        prob = torch.exp(log_prob)
        p_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1) # (N, H, W)

        # Calculate the modulating factor (1 - p_t)^gamma
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)

        # Calculate the final focal loss
        focal_loss = modulating_factor * ce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            # Ensure alpha is on the same device as the targets
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            # Expand alpha to match the shape of focal_loss for broadcasting
            alpha_t = self.alpha[targets] # Shape: (N, H, W)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss

# --- Data Preprocessing ---
# Define transforms common for torchvision models
# Using ImageNet stats as DeepLab is often pre-trained on it
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

# --- Dataset Definition ---
# Modified to use torchvision transforms and handle resizing
class SegmentationDataset(Dataset):
    def __init__(self, file_list, image_folder, label_folder, resize_shape, transform=None, calculate_weights=False):
        self.file_list = file_list
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.resize_shape = resize_shape # Tuple (H, W)
        self.class_counts = np.zeros(NUM_CLASSES)
        self.total_pixels = 0
        self.calculate_weights = calculate_weights # Flag to trigger weight calculation

        if self.calculate_weights:
            print("Calculating class frequencies for alpha weights...")
            self._calculate_class_frequencies()

    def __len__(self):
        return len(self.file_list)

    def _calculate_class_frequencies(self):
        for file_name in self.file_list:
            label_path = os.path.join(self.label_folder, file_name) + '.tif'
            try:
                label_img = Image.open(label_path).convert("L")
                label_np = np.array(label_img)
                unique, counts = np.unique(label_np, return_counts=True)
                for cls, count in zip(unique, counts):
                    if 0 <= cls < NUM_CLASSES:
                        self.class_counts[cls] += count
                self.total_pixels += label_np.size
            except Exception as e:
                print(f"Warning: Error processing {label_path} during weight calculation: {e}")
        print(f"Total pixels counted: {self.total_pixels}")
        print(f"Class counts: {self.class_counts}")

    def get_class_weights(self, beta=0.99):
        """Calculates inverse frequency weights, optionally smoothed."""
        if self.total_pixels == 0:
            print("Warning: Total pixels is zero. Cannot calculate weights.")
            return torch.ones(NUM_CLASSES)

        # Calculate inverse frequency
        class_freq = self.class_counts / self.total_pixels
        # Handle zero counts to avoid division by zero
        class_freq[class_freq == 0] = 1e-6
        
        # Simple Inverse Frequency
        # weights = 1.0 / class_freq

        # ENet Inverse Frequency Weighting (Smoothed)
        # weights = 1.0 / np.log(1.02 + class_freq) # 1.02 is a small smoothing factor

        # Median Frequency Balancing (often used in segmentation)
        median_freq = np.median(class_freq[self.class_counts > 0]) # Median of non-zero frequencies
        weights = median_freq / np.log(class_freq)

        # Normalize weights (optional, sometimes helps)
        weights /= weights.sum() 
        # weights *= NUM_CLASSES # Scale so average weight is 1

        print(f"Calculated alpha weights: {weights}")
        return torch.tensor(weights, dtype=torch.float)


    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        image_path = os.path.join(self.image_folder, file_name) + '.tif'
        label_path = os.path.join(self.label_folder, file_name) + '.tif'

        try:
            # Load image (convert to RGB as DeepLab expects 3 channels)
            image = Image.open(image_path).convert("RGB")
            # Load label (keep as single channel integer indices)
            label = Image.open(label_path).convert("L") # Use 'L' for single-channel integer labels

            # --- Resize Image and Label ---
            # Resize image using BILINEAR interpolation
            image = resize(image, self.resize_shape, interpolation=InterpolationMode.BILINEAR)
            # Resize label using NEAREST interpolation to avoid creating new class values
            label = resize(label, self.resize_shape, interpolation=InterpolationMode.NEAREST)

            # Apply other transforms (ToTensor, Normalize) to image
            if self.transform:
                image_tensor = self.transform(image)
            else:
                image_tensor = transforms.ToTensor()(image) # Fallback

            # Convert label to tensor (don't normalize labels)
            # Ensure label is Long type for loss function
            label_tensor = torch.tensor(np.array(label), dtype=torch.long)

        except Exception as e:
            print(f"Error loading or processing file: {file_name}")
            print(f"Image path: {image_path}")
            print(f"Label path: {label_path}")
            print(f"Error: {e}")
            # Return dummy data or raise error
            # Create dummy tensors of expected type and shape to avoid crashing DataLoader
            image_tensor = torch.zeros((3, *self.resize_shape), dtype=torch.float)
            label_tensor = torch.zeros(self.resize_shape, dtype=torch.long)
            # Or better: raise e

        return image_tensor, label_tensor

# --- Metric Calculation Functions (Keep as is from previous examples) ---
# Assumes logits and labels have the same spatial dimensions, which DeepLabV3+ usually ensures.
# If not, uncomment and adapt the F.interpolate lines within these functions.
def calculate_iou(outputs_logits, labels, num_classes):
    # Logits shape: (N, C, H, W), Labels shape: (N, H, W)
    # Check if upsampling is needed (unlikely for torchvision's DeepLabV3+)
    # if outputs_logits.shape[-2:] != labels.shape[-2:]:
    #     outputs_logits = F.interpolate(outputs_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
    preds = torch.argmax(outputs_logits, dim=1) # (N, H, W)

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
        union_counts.append(union.item() if union.item() > 0 else 1e-6) # Avoid division by zero

    return intersection_counts, union_counts

def calculate_recall(outputs_logits, labels, num_classes):
    # Logits shape: (N, C, H, W), Labels shape: (N, H, W)
    # Check if upsampling is needed
    # if outputs_logits.shape[-2:] != labels.shape[-2:]:
    #     outputs_logits = F.interpolate(outputs_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
    preds = torch.argmax(outputs_logits, dim=1) # (N, H, W)

    preds = preds.view(-1)
    labels = labels.view(-1)

    true_positives_counts = []
    actual_positives_counts = []
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)

        true_positives = torch.logical_and(pred_mask, label_mask).sum().float()
        total_ground_truth = label_mask.sum().float()

        true_positives_counts.append(true_positives.item())
        actual_positives_counts.append(total_ground_truth.item() if total_ground_truth.item() > 0 else 1e-6) # Avoid division by zero

    return true_positives_counts, actual_positives_counts


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

    # --- Create Datasets ---
    # Pass resize_size to the dataset
    # Create training dataset with flag to calculate weights if alpha is not provided
    should_calculate_weights = args.alpha is None
    image_folder = './data/05m_chips/slope/'
    label_folder = './data/05m_chips/labels/'

    train_dataset = SegmentationDataset(
        train_files, image_folder, label_folder,
        resize_shape=resize_size, transform=train_transform,
        calculate_weights=should_calculate_weights
    )
    # Validation and test datasets don't need to calculate weights
    val_dataset = SegmentationDataset(
        val_files, image_folder, label_folder,
        resize_shape=resize_size, transform=val_transform
    )
    test_dataset = SegmentationDataset(
        test_files, image_folder, label_folder,
        resize_shape=resize_size, transform=val_transform
    )
    print(f"Created datasets with {len(train_dataset)} train, {len(test_dataset)} test, {len(val_dataset)} val samples.")

    # --- Determine Alpha Weights ---
    alpha_weights = None
    if args.alpha:
        if len(args.alpha) == NUM_CLASSES:
            alpha_weights = torch.tensor(args.alpha, dtype=torch.float).to(device)
            print(f"Using provided alpha weights: {alpha_weights}")
        else:
            print(f"Error: Provided alpha list length ({len(args.alpha)}) does not match NUM_CLASSES ({NUM_CLASSES}). Using uniform weights.")
            alpha_weights = torch.ones(NUM_CLASSES, dtype=torch.float).to(device)
    elif should_calculate_weights:
        alpha_weights = train_dataset.get_class_weights().to(device)
        print(f"Using calculated alpha weights: {alpha_weights}")
    else:
         print("Using uniform alpha weights.")
         alpha_weights = torch.ones(NUM_CLASSES, dtype=torch.float).to(device) # Fallback to uniform


    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Created DataLoaders with batch size {batch_size}")

    # --- Create DeepLabV3+ model ---
    # Load pre-trained weights (using updated API)
    weights = DeepLabV3_ResNet50_Weights.DEFAULT # Corresponds to COCO pre-training
    model = deeplabv3_resnet50(weights=weights)

    # Modify the final classifier layer for the target number of classes
    # The ASPP classifier
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))
    # The auxiliary classifier (optional, but good practice to modify if it exists)
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
         model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=(1,1), stride=(1,1))

    model = model.to(device)
    print("DeepLabV3+ ResNet50 model loaded and moved to device.")
    # print(model) # Uncomment to see model structure

    # Loss and optimizer
    criterion = FocalLoss(alpha=alpha_weights, gamma=args.gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # AdamW often preferred

    # --- Training Loop ---
    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        print(f"\n--- Starting Epoch {epoch+1}/{num_epochs} ---")
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device) # shape: [batch_size, 3, H, W] (resized)
            labels = labels.to(device) # shape: [batch_size, H, W] (resized, LongTensor)

            optimizer.zero_grad()

            # Forward pass - DeepLabV3 returns a dictionary
            outputs = model(images)
            logits = outputs['out'] # Main output logits (N, C, H, W)

            # Calculate loss using Focal Loss
            loss = criterion(logits, labels)

            # Handle auxiliary loss if present and model is in training mode
            if 'aux' in outputs and model.training:
                aux_logits = outputs['aux']
                aux_loss = criterion(aux_logits, labels)
                loss = loss + 0.4 * aux_loss # Weight aux loss as is common practice

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1) % 10 == 0 or i == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}", flush=True)

        epoch_loss = running_loss / total_steps
        print(f" -- Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {epoch_loss:.4f}", flush=True)

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        total_intersections = [0] * NUM_CLASSES
        total_unions = [0] * NUM_CLASSES
        total_true_positives = [0] * NUM_CLASSES
        total_actual_positives = [0] * NUM_CLASSES
        num_val_steps = len(val_loader)

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                logits = outputs['out'] # Use main output for validation/inference

                # Calculate loss (using the same criterion, ignoring aux loss for validation)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                # --- Calculate Metrics ---
                # Pass main logits directly
                batch_intersections, batch_unions = calculate_iou(logits, labels, NUM_CLASSES)
                for cls in range(NUM_CLASSES):
                    total_intersections[cls] += batch_intersections[cls]
                    total_unions[cls] += batch_unions[cls]

                batch_true_positives, batch_actual_positives = calculate_recall(logits, labels, NUM_CLASSES)
                for cls in range(NUM_CLASSES):
                    total_true_positives[cls] += batch_true_positives[cls]
                    total_actual_positives[cls] += batch_actual_positives[cls]

        avg_val_loss = val_loss / num_val_steps
        avg_class_iou = []
        avg_class_recall = []
        for cls in range(NUM_CLASSES):
            iou = total_intersections[cls] / total_unions[cls] if total_unions[cls] > 1e-6 else 0.0
            avg_class_iou.append(iou)

            recall = total_true_positives[cls] / total_actual_positives[cls] if total_actual_positives[cls] > 1e-6 else 0.0
            avg_class_recall.append(recall)

        mean_iou = sum(avg_class_iou) / NUM_CLASSES
        mean_recall = sum(avg_class_recall) / NUM_CLASSES

        # Log results
        log_file_path = os.path.join(logdir, 'training.log')
        with open(log_file_path, 'a') as log_file:
            if epoch == 0:
                iou_headers = ','.join([f'iou_class{i}' for i in range(NUM_CLASSES)])
                recall_headers = ','.join([f'recall_class{i}' for i in range(NUM_CLASSES)])
                log_file.write(f'epoch,val_loss,mIoU,mRecall,{iou_headers},{recall_headers}\n')

            iou_values = ','.join([f'{iou:.4f}' for iou in avg_class_iou])
            recall_values = ','.join([f'{recall:.4f}' for recall in avg_class_recall])
            log_file.write(f"{epoch+1},{avg_val_loss:.4f},{mean_iou:.4f},{mean_recall:.4f},{iou_values},{recall_values}\n")

        end_time = time.time()
        print(f" -- Validation Loss: {avg_val_loss:.4f}", flush=True)
        print(f" -- Mean Validation IoU (mIoU): {mean_iou:.4f}", flush=True)
        print(f" -- Average Validation IoU per class: {[f'{iou:.4f}' for iou in avg_class_iou]}", flush=True)
        print(f" -- Mean Validation Recall: {mean_recall:.4f}", flush=True)
        print(f" -- Average Validation Recall per class: {[f'{recall:.4f}' for recall in avg_class_recall]}", flush=True)
        print(f" -- Epoch Time: {end_time - start_time:.2f} seconds\n", flush=True)

        # Save model checkpoint periodically
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(logdir, f'deeplab_focal_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")

    final_time = time.time()
    print(f"Training completed in {(final_time - begin_time) / 60:.2f} minutes", flush=True)

    # Save the final model
    final_model_path = os.path.join(logdir, 'deeplab_focal_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")