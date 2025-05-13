import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import time
import argparse
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import numpy as np # Added for potential weight calculation reference

# argparse
parser = argparse.ArgumentParser(description='SegFormer Training')
parser.add_argument('--logdir', type=str, default='logs_segformer', help='Directory to save logs')
parser.add_argument('--model_checkpoint', type=str, default='nvidia/segformer-b0-finetuned-ade-512-512', help='Hugging Face model checkpoint name')
args = parser.parse_args()
logdir = args.logdir
model_checkpoint = args.model_checkpoint

# Create log directory if it doesn't exist
os.makedirs(logdir, exist_ok=True)

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# costants
NUM_CLASSES = 3

# HPTs - Adjusted for typical Transformer training
num_epochs = 100
batch_size = 4
learning_rate = 5e-5

print(f"num_epochs: {num_epochs}, batch_size: {batch_size}, learning_rate: {learning_rate}")
print(f"Using SegFormer model: {model_checkpoint}")

# --- Feature Extractor ---
feature_extractor = SegformerFeatureExtractor.from_pretrained(
    model_checkpoint,
    do_reduce_labels=False
)
print("Feature extractor loaded.")

# --- Dataset Definition ---
class SegmentationDataset(Dataset):
    def __init__(self, file_list, image_folder, label_folder, feature_extractor):
        self.file_list = file_list
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        image_path = os.path.join(self.image_folder, file_name) + '.tif'
        label_path = os.path.join(self.label_folder, file_name) + '.tif'

        try:
            image = Image.open(image_path).convert("RGB")
            label = Image.open(label_path).convert("L")

        except Exception as e:
            print(f"Error loading file: {file_name}")
            print(f"Image path: {image_path}")
            print(f"Label path: {label_path}")
            print(f"Error: {e}")
            raise e

        inputs = self.feature_extractor(images=image, segmentation_maps=label, return_tensors="pt")

        image_tensor = inputs['pixel_values'].squeeze(0)
        label_tensor = inputs['labels'].squeeze(0)

        # Ensure label tensor is Long type for CrossEntropyLoss
        label_tensor = label_tensor.long()

        return image_tensor, label_tensor

# --- Metric Calculation Functions (Keep as is) ---
def calculate_iou(outputs_logits, labels, num_classes):
    # Upsample logits to match label size BEFORE argmax
    upsampled_logits = F.interpolate(outputs_logits,
                                     size=labels.shape[-2:], # Target H, W
                                     mode='bilinear',
                                     align_corners=False)
    preds = torch.argmax(upsampled_logits, dim=1)

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
        # Avoid division by zero if union is zero
        union_counts.append(union.item() if union.item() > 0 else 1e-6) # Add epsilon

    return intersection_counts, union_counts

def calculate_recall(outputs_logits, labels, num_classes):
    # Upsample logits to match label size BEFORE argmax
    upsampled_logits = F.interpolate(outputs_logits,
                                     size=labels.shape[-2:], # Target H, W
                                     mode='bilinear',
                                     align_corners=False)
    preds = torch.argmax(upsampled_logits, dim=1)

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
        # Avoid division by zero if no ground truth positives exist for the class
        actual_positives_counts.append(total_ground_truth.item() if total_ground_truth.item() > 0 else 1e-6) # Add epsilon

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

    # Create Datasets using the feature extractor
    image_folder = './data/05m_chips/slope/'
    label_folder = './data/05m_chips/labels/'
    train_dataset = SegmentationDataset(train_files, image_folder, label_folder, feature_extractor)
    test_dataset = SegmentationDataset(test_files, image_folder, label_folder, feature_extractor)
    val_dataset = SegmentationDataset(val_files, image_folder, label_folder, feature_extractor)
    print(f"Created datasets with {len(train_dataset)} train, {len(test_dataset)} test, {len(val_dataset)} val samples.")

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Created DataLoaders with batch size {batch_size}")

    # --- Define Class Weights for Loss ---
    # IMPORTANT: You MUST calculate these weights based on the pixel distribution
    # of classes in your TRAINING dataset.
    # Example placeholder weights (replace these!):
    # If class 0 is ~95%, class 1 is ~2.5%, class 2 is ~2.5%
    # Inverse frequencies: 1/0.95, 1/0.025, 1/0.025 -> approx [1.05, 40, 40]
    # Normalized (e.g., dividing by min weight): [1, ~38, ~38]
    # Or calculate pixel counts N0, N1, N2. Weights could be proportional to 1/Ni.
    # Let's use a placeholder, assuming class 0 is dominant, and classes 1 and 2 are rare.
    # Adjust these values based on your actual data distribution.
    # A common method is to set the majority class weight to 1 and scale others:
    # max_pixels = max(N0, N1, N2)
    # weights = [max_pixels / N0, max_pixels / N1, max_pixels / N2]
    # Then potentially normalize these, but CrossEntropyLoss doesn't require sum=1.
    # Using weights like [1.0, 20.0, 20.0] means misclassifying class 1 or 2
    # costs 20 times more than misclassifying class 0.
    class_weights = torch.tensor([1.0, 20.0, 20.0]).to(device) # REPLACE THESE WEIGHTS!
    print(f"\n--- Using Class Weights for Loss: {class_weights.tolist()} ---")

    # Loss and optimizer
    # Define the criterion using the calculated weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Create SegFormer model ---
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)
    print("SegFormer model loaded and moved to device.")


    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training Loop
    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        print(f"\n--- Starting Epoch {epoch+1}/{num_epochs} ---")
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device) # shape: [batch_size, H_label, W_label] (LongTensor)

            optimizer.zero_grad()

            # Forward pass: Get logits from the model
            # Use model() without 'labels' to get raw logits for calculating loss
            # with our custom weighted criterion.
            outputs = model(pixel_values=images) # outputs is a dictionary/object
            logits = outputs.logits # shape: [batch_size, num_classes, H_out, W_out]

            # Calculate loss using the weighted criterion
            # The criterion expects logits [N, C, H, W] and labels [N, H, W]
            # It internally handles reshaping logits to [N, C, H*W] and labels to [N, H*W]
            # and applies the weights.
            loss = criterion(logits, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1) % 10 == 0 or i == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}", flush=True)

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

                # Forward pass - get logits for calculating loss and metrics
                outputs = model(pixel_values=images)
                logits = outputs.logits

                # Calculate validation loss using the same weighted criterion
                loss = criterion(logits, labels)
                val_loss += loss.item()

                # --- Calculate Metrics ---
                # Pass logits to the metric functions
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
            # Ensure union and actual positives are not zero before division
            iou = total_intersections[cls] / (total_unions[cls] + 1e-6)
            avg_class_iou.append(iou)

            recall = total_true_positives[cls] / (total_actual_positives[cls] + 1e-6)
            avg_class_recall.append(recall)


        # Calculate mean IoU (mIoU) - average of class IoUs
        mean_iou = sum(avg_class_iou) / NUM_CLASSES
        mean_recall = sum(avg_class_recall) / NUM_CLASSES

        # Log results
        with open(os.path.join(logdir, 'training.log'), 'a') as log_file:
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
            checkpoint_path = os.path.join(logdir, f'segformer_epoch_{epoch+1}.pth')
            # Save the model's state_dict
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")


    final_time = time.time()
    print(f"Training completed in {(final_time - begin_time) / 60:.2f} minutes", flush=True)

    # Save the final model
    final_model_path = os.path.join(logdir, 'segformer_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")