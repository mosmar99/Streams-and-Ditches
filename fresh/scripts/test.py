import os
import time
import torch
import numpy as np
from datetime import datetime
from models.unet import add_padding, remove_padding
from utils.metrics import (
    calculate_tp_fp_fn_per_class,
    calculate_precision_recall_per_class,
    calculate_iou_per_class,
    calculate_dice_per_class,
    calculate_overall_pixel_accuracy
)

def test_unet(model, test_loader, criterion, device, num_classes, checkpoint_path=None, 
              logdir='test_results/', output_dir=None):
    """
    Tests a U-Net model on a given test dataset.

    Args:
        model (torch.nn.Module): The U-Net model to test.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): The loss function (e.g., TverskyLoss, CrossEntropyLoss).
        device (torch.device): The device to run the model on (e.g., 'cuda', 'cpu').
        num_classes (int): The number of segmentation classes.
        checkpoint_path (str, optional): Path to the trained model checkpoint. If None, uses the model as is.
        logdir (str, optional): Directory to save test logs.
        output_dir (str, optional): Directory to save predicted masks (if desired).
    """
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    model.to(device)
    model.eval() 

    total_loss = 0.0
    all_pred_masks_flat = []
    all_true_masks_flat = []
    
    log_file_path = os.path.join(logdir, 'testing_metrics.log')
    with open(log_file_path, 'w') as f:
        f.write(f"Testing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}, Model: {model.__class__.__name__}\n")
        if checkpoint_path:
            f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Batch Size: {test_loader.batch_size}\n")
        f.write(f"Loss Function: {criterion.__class__.__name__}\n")
        f.write("Batch, BatchLoss\n") # Header for per-batch loss

    print(f"Starting testing on {len(test_loader.dataset)} images...")
    start_time = time.time()

    with torch.no_grad(): # Disable gradient calculations for testing
        for i, (images, masks) in enumerate(test_loader):
            current_batch_num = i + 1
            images = images.to(device)
            masks = masks.to(device) # Ground truth masks

            images_padded, padding_info = add_padding(images)
            images_padded = images_padded.to(device)
            
            outputs_padded_logits = model(images_padded)
            outputs_logits = remove_padding(outputs_padded_logits, padding_info)
            outputs_logits = outputs_logits.to(device)

            loss = criterion(outputs_logits, masks)
            current_batch_loss = loss.item()
            total_loss += current_batch_loss * images.size(0)

            pred_mask_indices = torch.argmax(outputs_logits, dim=1) # Shape: [N, H, W]

            all_pred_masks_flat.append(pred_mask_indices.cpu().view(-1))
            all_true_masks_flat.append(masks.cpu().view(-1))
            
            if current_batch_num % 10 == 0:
                print(f"  Batch [{current_batch_num}/{len(test_loader)}], Loss: {current_batch_loss:.4f}", flush=True)
            
            with open(log_file_path, 'a') as f:
                f.write(f"{current_batch_num}, {current_batch_loss:.6f}\n")
                f.flush()

            # Optional: Save predicted masks as images
            if output_dir:
                for k in range(pred_mask_indices.size(0)):
                    pred_img = TF.to_pil_image(pred_mask_indices[k].byte().cpu() * (255 // (num_classes -1 if num_classes > 1 else 1) )) 
                    image_idx_in_batch = i * test_loader.batch_size + k 
                    pred_img.save(os.path.join(output_dir, f"pred_mask_{image_idx_in_batch:04d}.png"))

    end_time = time.time()
    avg_test_loss = total_loss / len(test_loader.dataset)
    
    print(f"--- Testing complete. Average Test Loss: {avg_test_loss:.4f} ---", flush=True)
    print(f"--- Time taken for testing: {end_time - start_time:.2f} seconds ---", flush=True)

    # Concatenate all flattened masks for global metric calculation
    all_pred_masks_tensor_flat = torch.cat(all_pred_masks_flat, dim=0)
    all_true_masks_tensor_flat = torch.cat(all_true_masks_flat, dim=0)

    # Calculate metrics
    tps, fps, fns = calculate_tp_fp_fn_per_class(all_pred_masks_tensor_flat, all_true_masks_tensor_flat, num_classes)
    precisions, recalls = calculate_precision_recall_per_class(tps, fps, fns)
    ious = calculate_iou_per_class(tps, fps, fns)
    dices = calculate_dice_per_class(tps, fps, fns)
    pixel_acc = calculate_overall_pixel_accuracy(all_pred_masks_tensor_flat, all_true_masks_tensor_flat)

    mean_iou = np.mean(ious)
    mean_dice = np.mean(dices)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)

    print(f"Overall Pixel Accuracy: {pixel_acc:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    
    with open(log_file_path, 'a') as f:
        f.write(f"\n--- Overall Metrics ---\n")
        f.write(f"Testing finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Average Test Loss: {avg_test_loss:.6f}\n")
        f.write(f"Overall Pixel Accuracy: {pixel_acc:.6f}\n")
        f.write(f"Mean IoU: {mean_iou:.6f}\n")
        f.write(f"Mean Dice Score: {mean_dice:.6f}\n")
        f.write(f"Mean Precision: {mean_precision:.6f}\n")
        f.write(f"Mean Recall: {mean_recall:.6f}\n")
        f.write("\n--- Per-Class Metrics ---\n")
        header = "Class | Precision | Recall    | IoU       | Dice\n"
        f.write(header)
        print("\n--- Per-Class Metrics ---")
        print(header.strip())
        for c in range(num_classes):
            class_metrics_str = f"{c:<5} | {precisions[c]:<9.4f} | {recalls[c]:<9.4f} | {ious[c]:<9.4f} | {dices[c]:<9.4f}\n"
            f.write(class_metrics_str)
            print(class_metrics_str.strip())
        f.flush()
        
    print("UNet testing finished.", flush=True)
    # Return a dictionary of metrics for potential further use
    metrics_summary = {
        "avg_test_loss": avg_test_loss,
        "overall_pixel_accuracy": pixel_acc,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "per_class_iou": ious,
        "per_class_dice": dices,
        "per_class_precision": precisions,
        "per_class_recall": recalls
    }
    return metrics_summary