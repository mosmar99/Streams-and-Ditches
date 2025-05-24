# workspace/scripts/train.py
import os
import time 
import torch
import numpy as np
from datetime import datetime
from models.unet import add_padding, remove_padding

class MinCheckpoint():
    def __init__(self, checkpoint_dir, model_name='unet_model_'):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, model_state_dict, epoch):
        if epoch in [25, 50, 75, 100, 125, 150, 175, 200]:
            print(f" -- Updated Checkpoint: Saved model at {epoch} epochs.", flush=True)
            checkpoint_filename = f"{self.model_name}_epoch{epoch}.pth"
            torch.save(model_state_dict, os.path.join(self.checkpoint_dir, checkpoint_filename))

def train_unet(model, train_loader, criterion, optimizer, num_epochs, device, 
               logdir='checkpoints/', model_name='unet_ckpt.pth', weight_decay=5e-7, dropout=0.05, patch_size=128):
    
    checkpoint_dir = os.path.join(logdir, 'checkpoints')
    checkpoint_saver = MinCheckpoint(checkpoint_dir=checkpoint_dir, model_name=model_name)
    
    model.to(device)
    
    log_file_path = os.path.join(logdir, 'training.log')
    with open(log_file_path, 'a') as f:
        f.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}, Num Epochs: {num_epochs}, Model: {model.__class__.__name__}\n")
        f.write(f"Batch Size: {train_loader.batch_size}, Learning Rate: {optimizer.param_groups[0]['lr']}\n")
        f.write(f"Loss Function: {criterion.__class__.__name__}, Optimizer: {optimizer.__class__.__name__}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write(f"Dropout: {dropout}\n")
        f.write(f"Patch Size: {patch_size}\n")
        f.write("Epoch, AvgTrainLoss, \n")
    
    total_batches_per_epoch = len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        start_time = time.time()
        for i, (images, masks, _) in enumerate(train_loader):
            current_batch_num = i + 1
            images = images.to(device)
            masks = masks.to(device)
            images_padded, padding_info = add_padding(images)
            images_padded = images_padded.to(device)
            optimizer.zero_grad()
            outputs_padded_logits = model(images_padded)
            outputs_logits = remove_padding(outputs_padded_logits, padding_info)
            outputs_logits = outputs_logits.to(device)
            loss = criterion(outputs_logits, masks)
            loss.backward()
            optimizer.step()
            current_batch_loss = loss.item()
            running_loss += current_batch_loss * images.size(0)
            if current_batch_num % 100 == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}], Batch [{current_batch_num}/{total_batches_per_epoch}], Loss: {current_batch_loss:.4f}", flush=True)

        end_time = time.time()
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"--- Epoch [{epoch+1}/{num_epochs}] complete. Average Training Loss: {epoch_loss:.4f} ---", flush=True)
        print(f"--- Time taken for epoch: {end_time - start_time:.2f} seconds ---", flush=True)
        checkpoint_saver.save(model.state_dict(), epoch+1)

        with open(log_file_path, 'a') as f:
            f.write(f"{epoch+1}, {epoch_loss:.6f}, \n")
            f.flush()

    print("UNet training finished.", flush=True)
    with open(log_file_path, 'a') as f:
        f.write(f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.flush()
    return model