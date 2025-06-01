# import math
# import torch
# import torchvision
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image
# import os
# from torch.utils.data import Dataset, DataLoader
# import time
# import numpy as np
# import argparse

# parser = argparse.ArgumentParser(description='UNet Training')
# parser.add_argument('--logdir', type=str, default='logs', help='Directory to save logs')
# args = parser.parse_args()
# logdir = args.logdir

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NUM_CLASSES = 3

# num_epochs = 100
# batch_size = 6
# learning_rate = 0.001

# print(f"num_epochs: {num_epochs}, batch_size: {batch_size}, learning_rate: {learning_rate}")

# def double_conv(in_c, out_c, dropout_prob=0.2):
#     conv = nn.Sequential(
#         nn.Conv2d(in_c, out_c, kernel_size=3, padding='same', bias=False),
#         nn.BatchNorm2d(out_c),
#         nn.ReLU(inplace=True),
#         # nn.Conv2d(out_c, out_c, kernel_size=3, padding='same', bias=False),
#         # nn.BatchNorm2d(out_c),
#         # nn.ReLU(inplace=True),
#         nn.Dropout(p=dropout_prob)
#     )
#     return conv

# def crop_img(og_tensor, target_tensor):
#     target_size_h, target_size_w = target_tensor.size()[2], target_tensor.size()[3]
#     og_size_h, og_size_w = og_tensor.size()[2], og_tensor.size()[3]

#     diff_h = og_size_h - target_size_h
#     diff_w = og_size_w - target_size_w

#     delta_h_top = diff_h // 2
#     delta_h_bottom = diff_h - delta_h_top
#     delta_w_left = diff_w // 2
#     delta_w_right = diff_w - delta_w_left
    
#     return og_tensor[:, :, delta_h_top:og_size_h-delta_h_bottom, delta_w_left:og_size_w-delta_w_right]


# def add_padding(inputs):
#     _, _, height, width = inputs.size()

#     # Pad height and width to be a multiple of 2^(number of pooling layers)
#     # For new depth (256 max filters), there are 3 pooling layers. So, divisible by 2^3 = 8.
#     # To be safe and general, let's aim for divisibility by 16 (2^4) if max_depth was 4, or 2^3 for max_depth 3.
#     # Current deepest is X3_0, meaning 3 poolings. So size should be mult of 8.
    
#     num_pool_layers = 3 # Since X0_0 -> X1_0 -> X2_0 -> X3_0 (bottleneck)
#     divisor = 2**num_pool_layers # 2^3 = 8

#     pad_h_total = 0
#     if height % divisor != 0:
#         pad_h_total = divisor - (height % divisor)

#     pad_w_total = 0
#     if width % divisor != 0:
#         pad_w_total = divisor - (width % divisor)

#     pad_top = pad_h_total // 2
#     pad_bottom = pad_h_total - pad_top
#     pad_left = pad_w_total // 2
#     pad_right = pad_w_total - pad_left

#     padding = (pad_left, pad_right, pad_top, pad_bottom) # (left, right, top, bottom) for F.pad
#     padded_inputs = nn.functional.pad(inputs, padding, mode='constant', value=0)

#     return padded_inputs, padding

# def remove_padding(tensor, padding):
#     pad_left, pad_right, pad_top, pad_bottom = padding

#     _, _, H, W = tensor.shape
    
#     # Slicing end index is exclusive. If pad_bottom is 0, we want to go up to H.
#     # So, H - 0 = H. If pad_bottom is >0, H - pad_bottom.
#     end_h = H - pad_bottom
#     end_w = W - pad_right
    
#     return tensor[:, :, pad_top:end_h, pad_left:end_w]


# class MinCheckpoint():
#     def __init__(self, logdir):
#         self.min_loss = np.inf
#         self.logdir = logdir

#     def save(self, model_to_save, loss):
#         if loss < self.min_loss:
#             print(f" -- Updated Checkpoint: {self.min_loss} > {loss}", flush=True)
#             self.min_loss = loss
#             torch.save(model_to_save.state_dict(), os.path.join(self.logdir, 'unet_model_ckpt.pth'))
            

# class UNet(nn.Module): # This is UNet++ with reduced depth
#     def __init__(self):
#         super(UNet, self).__init__()

#         in_channels = 1
#         n_classes = NUM_CLASSES
#         init_features = 32
#         self.deep_supervision = False 

#         # New depth: deepest layer 256 filters. nb_filter = [32, 64, 128, 256]
#         nb_filter = [init_features, init_features*2, init_features*4, init_features*8]

#         self.pool = nn.MaxPool2d(2, 2)
#         self.ups = nn.ModuleList()
#         # Number of upsampling layers = len(nb_filter) - 1 = 4 - 1 = 3
#         for i in range(len(nb_filter) - 1): 
#             self.ups.append(nn.ConvTranspose2d(nb_filter[i+1], nb_filter[i], kernel_size=2, stride=2))

#         # Encoder (nodes X_i_0)
#         self.conv0_0 = double_conv(in_channels, nb_filter[0])    # X_0_0
#         self.conv1_0 = double_conv(nb_filter[0], nb_filter[1])  # X_1_0
#         self.conv2_0 = double_conv(nb_filter[1], nb_filter[2])  # X_2_0
#         self.conv3_0 = double_conv(nb_filter[2], nb_filter[3])  # X_3_0 (Bottleneck)

#         # Nodes X_i_j for j > 0
#         # Column j=1
#         self.conv0_1 = double_conv(nb_filter[0] + nb_filter[0], nb_filter[0]) # Inputs: X0_0, Up(X1_0)
#         self.conv1_1 = double_conv(nb_filter[1] + nb_filter[1], nb_filter[1]) # Inputs: X1_0, Up(X2_0)
#         self.conv2_1 = double_conv(nb_filter[2] + nb_filter[2], nb_filter[2]) # Inputs: X2_0, Up(X3_0)
        
#         # Column j=2
#         self.conv0_2 = double_conv(nb_filter[0]*2 + nb_filter[0], nb_filter[0]) # Inputs: X0_0, X0_1, Up(X1_1)
#         self.conv1_2 = double_conv(nb_filter[1]*2 + nb_filter[1], nb_filter[1]) # Inputs: X1_0, X1_1, Up(X2_1)
        
#         # Column j=3
#         self.conv0_3 = double_conv(nb_filter[0]*3 + nb_filter[0], nb_filter[0]) # Inputs: X0_0, X0_1, X0_2, Up(X1_2)

#         # Final output layer (output from X_0_3)
#         if self.deep_supervision:
#             # If deep_supervision were true, we'd have multiple output layers
#             self.final1 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1) # from X0_1
#             self.final2 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1) # from X0_2
#             self.final3 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1) # from X0_3
#         else:
#             self.out = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)


#     def forward(self, image):
#         # Encoder path (X_i_0)
#         x0_0 = self.conv0_0(image)
#         x1_0 = self.conv1_0(self.pool(x0_0))
#         x2_0 = self.conv2_0(self.pool(x1_0))
#         x3_0 = self.conv3_0(self.pool(x2_0)) # Bottleneck

#         # Decoder / Skip path nodes
#         # Column j=1
#         # Upsample X1_0 (nb_filter[1]) to match X0_0 (nb_filter[0]) channels using self.ups[0]
#         up1_0 = self.ups[0](x1_0) 
#         x0_1 = self.conv0_1(torch.cat([crop_img(x0_0, up1_0), up1_0], 1))

#         # Upsample X2_0 (nb_filter[2]) to match X1_0 (nb_filter[1]) channels using self.ups[1]
#         up2_0 = self.ups[1](x2_0)
#         x1_1 = self.conv1_1(torch.cat([crop_img(x1_0, up2_0), up2_0], 1))

#         # Upsample X3_0 (nb_filter[3]) to match X2_0 (nb_filter[2]) channels using self.ups[2]
#         up3_0 = self.ups[2](x3_0)
#         x2_1 = self.conv2_1(torch.cat([crop_img(x2_0, up3_0), up3_0], 1))
        
#         # Column j=2
#         # Upsample X1_1 (nb_filter[1]) to match X0_ (nb_filter[0]) channels using self.ups[0]
#         up1_1 = self.ups[0](x1_1) 
#         x0_2 = self.conv0_2(torch.cat([crop_img(x0_0, up1_1), crop_img(x0_1, up1_1), up1_1], 1))

#         # Upsample X2_1 (nb_filter[2]) to match X1_ (nb_filter[1]) channels using self.ups[1]
#         up2_1 = self.ups[1](x2_1)
#         x1_2 = self.conv1_2(torch.cat([crop_img(x1_0, up2_1), crop_img(x1_1, up2_1), up2_1], 1))
        
#         # Column j=3
#         # Upsample X1_2 (nb_filter[1]) to match X0_ (nb_filter[0]) channels using self.ups[0]
#         up1_2 = self.ups[0](x1_2)
#         x0_3 = self.conv0_3(torch.cat([crop_img(x0_0, up1_2), crop_img(x0_1, up1_2), crop_img(x0_2, up1_2), up1_2], 1))

#         if self.deep_supervision:
#             output1 = self.final1(x0_1)
#             output2 = self.final2(x0_2)
#             output3 = self.final3(x0_3)
#             return (output1 + output2 + output3) / 3 # Example: average
#         else:
#             # Single output from X_0_3 node
#             output = self.out(x0_3)
#             return output

#     def predict(self, images):
#         images_padded, padding = add_padding(images)
#         images_padded = images_padded.to(device)

#         outputs = self(images_padded) 
#         outputs_cropped = remove_padding(outputs, padding)
#         return outputs_cropped

#     def _train(self, train_loader, num_epochs, criterion, optimizer, epoch, total_steps):
#         self.train() 
#         running_loss = 0.0
#         for i, (images, labels) in enumerate(train_loader):
#             outputs_cropped = self.predict(images) 
#             squeezed_labels = labels.squeeze(1).long().to(device)

#             loss = criterion(outputs_cropped, squeezed_labels) + 0.01 * torch.mean(outputs_cropped ** 2)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * images.size(0)

#             if (i+1) % 46 == 0:
#                 print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item()}", flush=True)

#         epoch_loss = running_loss / len(train_dataset) 
#         print(f" -- Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {epoch_loss:.4f}", flush=True)
#         return running_loss 

#     def _validate(self, val_loader, criterion, epoch):
#         self.eval() 
#         val_loss = 0.0

#         total_intersections = [0] * NUM_CLASSES
#         total_unions = [0] * NUM_CLASSES
#         total_true_positives_recall = [0] * NUM_CLASSES
#         total_actual_positives_in_label = [0] * NUM_CLASSES

#         total_tp_f1 = [0] * NUM_CLASSES
#         total_fp_f1 = [0] * NUM_CLASSES
#         total_fn_f1 = [0] * NUM_CLASSES

#         total_tp_mcc = [0] * NUM_CLASSES
#         total_tn_mcc = [0] * NUM_CLASSES
#         total_fp_mcc = [0] * NUM_CLASSES
#         total_fn_mcc = [0] * NUM_CLASSES

#         total_ditch_predicted_as_stream = 0
#         total_predicted_ditches = 0

#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images_padded, padding = add_padding(images) 
#                 labels_padded, _ = add_padding(labels)
#                 images_padded = images_padded.to(device)
#                 squeezed_labels_padded = labels_padded.squeeze(1).long().to(device)

#                 outputs = self(images_padded) 
#                 loss = criterion(outputs, squeezed_labels_padded) 
#                 val_loss += loss.item() * images_padded.size(0)
                
#                 preds = torch.argmax(outputs, dim=1)
#                 preds_flat = preds.view(-1)
#                 labels_flat = squeezed_labels_padded.view(-1)

#                 batch_intersections, batch_unions = calculate_iou(outputs, squeezed_labels_padded, NUM_CLASSES)
#                 for cls in range(NUM_CLASSES):
#                     total_intersections[cls] += batch_intersections[cls]
#                     total_unions[cls] += batch_unions[cls]

#                 batch_tp_recall, batch_ap_recall = calculate_recall(outputs, squeezed_labels_padded, NUM_CLASSES)
#                 for cls in range(NUM_CLASSES):
#                     total_true_positives_recall[cls] += batch_tp_recall[cls]
#                     total_actual_positives_in_label[cls] += batch_ap_recall[cls]

#                 batch_tp_f1_val, batch_fp_f1_val, batch_fn_f1_val = calculate_f1_components(outputs, squeezed_labels_padded, NUM_CLASSES)
#                 for cls in range(NUM_CLASSES):
#                     total_tp_f1[cls] += batch_tp_f1_val[cls]
#                     total_fp_f1[cls] += batch_fp_f1_val[cls]
#                     total_fn_f1[cls] += batch_fn_f1_val[cls]

#                 batch_tp_mcc_val, batch_tn_mcc_val, batch_fp_mcc_val, batch_fn_mcc_val = calculate_mcc_components(outputs, squeezed_labels_padded, NUM_CLASSES)
#                 for cls in range(NUM_CLASSES):
#                     total_tp_mcc[cls] += batch_tp_mcc_val[cls]
#                     total_tn_mcc[cls] += batch_tn_mcc_val[cls]
#                     total_fp_mcc[cls] += batch_fp_mcc_val[cls]
#                     total_fn_mcc[cls] += batch_fn_mcc_val[cls]

#                 ditch_pred_stream_actual_mask = (preds_flat == 1) & (labels_flat == 2)
#                 total_ditch_predicted_as_stream += ditch_pred_stream_actual_mask.sum().item()
#                 total_predicted_ditches += (preds_flat == 1).sum().item()

#         val_loss /= len(val_dataset) 
#         avg_class_iou = []
#         avg_class_recall = []
#         avg_class_f1 = []
#         avg_class_mcc = []

#         for cls in range(NUM_CLASSES):
#             iou = total_intersections[cls] / total_unions[cls] if total_unions[cls] > 0 else 0.0
#             avg_class_iou.append(iou)

#             recall_val = total_true_positives_recall[cls] / total_actual_positives_in_label[cls] if total_actual_positives_in_label[cls] > 0 else 0.0
#             avg_class_recall.append(recall_val)

#             tp_f1, fp_f1, fn_f1 = total_tp_f1[cls], total_fp_f1[cls], total_fn_f1[cls]
#             precision_f1 = tp_f1 / (tp_f1 + fp_f1) if (tp_f1 + fp_f1) > 0 else 0.0
#             recall_f1_metric = tp_f1 / (tp_f1 + fn_f1) if (tp_f1 + fn_f1) > 0 else 0.0 
#             f1_score = 2 * (precision_f1 * recall_f1_metric) / (precision_f1 + recall_f1_metric) if (precision_f1 + recall_f1_metric) > 0 else 0.0
#             avg_class_f1.append(f1_score)

#             tp, tn, fp, fn = total_tp_mcc[cls], total_tn_mcc[cls], total_fp_mcc[cls], total_fn_mcc[cls]
#             numerator = (tp * tn) - (fp * fn)
#             denominator_val_sqrt_term = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
#             denominator_val_sqrt = math.sqrt(denominator_val_sqrt_term) if denominator_val_sqrt_term > 0 else 0
#             mcc = 0.0 if denominator_val_sqrt == 0 else numerator / denominator_val_sqrt
#             avg_class_mcc.append(mcc)
            
#         frac_pred_ditch_actual_stream = total_ditch_predicted_as_stream / total_predicted_ditches if total_predicted_ditches > 0 else 0.0

#         if not os.path.exists(logdir):
#             os.makedirs(logdir)

#         log_header = 'epoch,val_loss'
#         log_values = f"{epoch + 1},{val_loss:.4f}"

#         recalls = [f'{avg_class_recall[i]:.4f}' for i in range(NUM_CLASSES)]
#         f1_scores_log = [f'{avg_class_f1[i]:.4f}' for i in range(NUM_CLASSES)] 
#         mccs = [f'{avg_class_mcc[i]:.4f}' for i in range(NUM_CLASSES)]
#         ious = [f'{avg_class_iou[i]:.4f}' for i in range(NUM_CLASSES)]

#         log_header += ''.join([f',recall_class{i}' for i in range(NUM_CLASSES)])
#         log_values += ''.join([f',{recalls[i]}' for i in range(NUM_CLASSES)])
#         log_header += ''.join([f',f1_class{i}' for i in range(NUM_CLASSES)])
#         log_values += ''.join([f',{f1_scores_log[i]}' for i in range(NUM_CLASSES)])
#         log_header += ''.join([f',mcc_class{i}' for i in range(NUM_CLASSES)])
#         log_values += ''.join([f',{mccs[i]}' for i in range(NUM_CLASSES)])
#         log_header += ''.join([f',iou_class{i}' for i in range(NUM_CLASSES)])
#         log_values += ''.join([f',{ious[i]}' for i in range(NUM_CLASSES)])
#         log_header += ',frac_pred_ditch_actual_stream'
#         log_values += f',{frac_pred_ditch_actual_stream:.4f}'

#         with open(f'{logdir}/training.log', 'a') as log_file:
#             if epoch == 0:
#                 log_file.write(log_header + '\n')
#             log_file.write(log_values + '\n')

#         print(f" -- Validation Loss: {val_loss:.4f}", flush=True)
#         print(f" -- Average Validation IoU per class:     {avg_class_iou}", flush=True)
#         print(f" -- Average Validation Recall per class:  {avg_class_recall}", flush=True)
#         print(f" -- Average Validation F1 per class:      {avg_class_f1}", flush=True)
#         print(f" -- Average Validation MCC per class:     {avg_class_mcc}", flush=True)
#         print(f" -- Validation Fraction Pred Ditch Actual Stream: {frac_pred_ditch_actual_stream:.4f}", flush=True)

#         checkpoint_handler.save(self, val_loss) 
        
#     def fit(self, train_loader, val_loader, num_epochs, criterion, optimizer):
#         total_steps = len(train_loader)
#         for epoch in range(num_epochs):
#             start_time = time.time()
#             self._train(train_loader, num_epochs, criterion, optimizer, epoch, total_steps)
#             self._validate(val_loader, criterion, epoch)
#             end_time = time.time()
#             print(f" -- Time: {end_time - start_time:.2f} seconds", flush=True)


# class UNetDataset(Dataset):
#     def __init__(self, file_list, image_folder, label_folder):
#         self.file_list = file_list
#         self.image_folder = image_folder
#         self.label_folder = label_folder
#         self.transform = transforms.ToTensor()

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         file_name = self.file_list[idx]
#         image_path = os.path.join(self.image_folder, file_name) + '.tif'
#         label_path = os.path.join(self.label_folder, file_name) + '.tif'

#         image = Image.open(image_path).convert("F")
#         label = Image.open(label_path).convert("F")

#         image_tensor = self.transform(image)
#         label_tensor = self.transform(label)

#         return image_tensor, label_tensor

# def calculate_iou(outputs_logits, labels, num_classes):
#     preds = torch.argmax(outputs_logits, dim=1)
#     preds = preds.view(-1)
#     labels = labels.view(-1)
#     intersection_counts = []
#     union_counts = []
#     for cls in range(num_classes):
#         pred_mask = (preds == cls)
#         label_mask = (labels == cls)

#         intersection = torch.logical_and(pred_mask, label_mask).sum().float()
#         union = torch.logical_or(pred_mask, label_mask).sum().float()

#         intersection_counts.append(intersection.item())
#         union_counts.append(union.item())

#     return intersection_counts, union_counts

# def calculate_recall(outputs_logits, labels, num_classes):
#     preds = torch.argmax(outputs_logits, dim=1)
#     preds = preds.view(-1)
#     labels = labels.view(-1)

#     batch_true_positives = []
#     batch_actual_positives = []

#     for cls in range(num_classes):
#         pred_mask = (preds == cls)
#         label_mask = (labels == cls)

#         tp = torch.logical_and(pred_mask, label_mask).sum().item()
#         batch_true_positives.append(tp)

#         actual_positives = label_mask.sum().item()
#         batch_actual_positives.append(actual_positives)

#     return batch_true_positives, batch_actual_positives

# def calculate_f1_components(outputs_logits, labels, num_classes):
#     preds = torch.argmax(outputs_logits, dim=1).view(-1)
#     labels = labels.view(-1)

#     batch_tp = []
#     batch_fp = []
#     batch_fn = []

#     for cls in range(num_classes):
#         pred_mask_cls = (preds == cls)
#         label_mask_cls = (labels == cls)

#         tp = torch.logical_and(pred_mask_cls, label_mask_cls).sum().item()
#         fp = torch.logical_and(pred_mask_cls, ~label_mask_cls).sum().item()
#         fn = torch.logical_and(~pred_mask_cls, label_mask_cls).sum().item()

#         batch_tp.append(tp)
#         batch_fp.append(fp)
#         batch_fn.append(fn)

#     return batch_tp, batch_fp, batch_fn

# def calculate_mcc_components(outputs_logits, labels, num_classes):
#     preds = torch.argmax(outputs_logits, dim=1).view(-1)
#     labels_flat = labels.view(-1)

#     batch_tp = []
#     batch_tn = []
#     batch_fp = []
#     batch_fn = []

#     for cls in range(num_classes):
#         pred_is_cls = (preds == cls)
#         label_is_cls = (labels_flat == cls)

#         pred_is_not_cls = ~pred_is_cls
#         label_is_not_cls = ~label_is_cls

#         tp = torch.logical_and(pred_is_cls, label_is_cls).sum().item()
#         tn = torch.logical_and(pred_is_not_cls, label_is_not_cls).sum().item()
#         fp = torch.logical_and(pred_is_cls, label_is_not_cls).sum().item()
#         fn = torch.logical_and(pred_is_not_cls, label_is_cls).sum().item()

#         batch_tp.append(tp)
#         batch_tn.append(tn)
#         batch_fp.append(fp)
#         batch_fn.append(fn)

#     return batch_tp, batch_tn, batch_fp, batch_fn

# def load_and_process_data():
#     def read_file_list(file_path):
#         with open(file_path, 'r') as f:
#             return [line.strip() for line in f.readlines()]

#     test_fnames_path = './data/mapio_folds/f1/test_files.dat'
#     train_fnames_path = './data/mapio_folds/f1/train_files.dat'
#     val_fnames_path = './data/mapio_folds/f1/val_files.dat'
#     print("Beginning to load data...")
#     train_files = read_file_list(train_fnames_path)
#     test_files = read_file_list(test_fnames_path)
#     val_files = read_file_list(val_fnames_path)

#     image_folder = './data/05m_chips/slope/'
#     label_folder = './data/05m_chips/labels/'
#     train_dataset_obj = UNetDataset(train_files, image_folder, label_folder) 
#     test_dataset_obj = UNetDataset(test_files, image_folder, label_folder)
#     val_dataset_obj = UNetDataset(val_files, image_folder, label_folder)
#     print(f"Created datasets with {len(train_dataset_obj)} train, {len(test_dataset_obj)} test, {len(val_dataset_obj)} val samples.")

#     train_loader = DataLoader(dataset=train_dataset_obj, batch_size=batch_size, shuffle=True, num_workers=15, pin_memory=True)
#     test_loader = DataLoader(dataset=test_dataset_obj, batch_size=batch_size, shuffle=False, num_workers=15, pin_memory=True)
#     val_loader = DataLoader(dataset=val_dataset_obj, batch_size=batch_size, shuffle=False, num_workers=15, pin_memory=True)
#     print(f"Created DataLoaders with batch size {batch_size}")

#     return train_loader, test_loader, val_loader, train_dataset_obj, test_dataset_obj, val_dataset_obj

# class WeightedMSE(nn.Module):

#     '''Weighted mean square error loss'''

#     def __init__(self, weights=None, device='cuda'):
#         '''Setup loss

#         Parameters
#         ----------
#         weights : List of class weights, optional
#         device : Select device to run on, optional
#         '''
#         nn.Module.__init__(self)
#         if weights is not None:
#              self._weights = torch.from_numpy(np.array(weights)).float().to(device) 
#         else:
#             self._weights = None


#     def forward(self, inputs, targets):
#         '''Compute loss for given inputs and targets

#         Parameters
#         ----------
#         inputs : Predicted output
#         targets : Target output

#         Returns
#         -------
#         Weighted MSE

#         '''
#         if self._weights is not None:
#             mse = ((inputs - targets) ** 2).mean(dim=(0, 2, 3)) 
#             mse = (self._weights * mse).sum()
#         else:
#             mse = ((inputs - targets) ** 2).mean()

#         return mse
    
# if __name__ == "__main__":

#     begin_time = time.time()
#     checkpoint_handler = MinCheckpoint(logdir)
#     train_loader, test_loader, val_loader, train_dataset, test_dataset, val_dataset = load_and_process_data()


#     model = UNet().to(device)

#     class_weights = torch.tensor([1.0, 100.0, 900.0], device=device)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.005)

#     model.fit(train_loader, val_loader, num_epochs, criterion, optimizer)

#     final_time = time.time()
#     print(f"Training completed in {(final_time - begin_time) / 60:.2f} minutes", flush=True)

#     torch.save(model.state_dict(), os.path.join(logdir, 'final_unet_model_ckpt.pth'))
#     print("Model saved as unet_model.pth")

#  # --------------- START OF TEST SECTION ---------------
#     print("\nStarting testing phase with the best saved model...")

#     best_model_path = os.path.join(logdir, 'unet_model_ckpt.pth')
#     if os.path.exists(best_model_path):
#         print(f"Loading best model from: {best_model_path}")
#         test_model = UNet().to(device) 
#         test_model.load_state_dict(torch.load(best_model_path, map_location=device))
#         test_model.eval()

#         test_loss = 0.0
#         test_total_intersections = [0] * NUM_CLASSES
#         test_total_unions = [0] * NUM_CLASSES
#         test_total_true_positives_recall = [0] * NUM_CLASSES
#         test_total_actual_positives_in_label = [0] * NUM_CLASSES
#         test_total_tp_f1 = [0] * NUM_CLASSES
#         test_total_fp_f1 = [0] * NUM_CLASSES
#         test_total_fn_f1 = [0] * NUM_CLASSES
#         test_total_tp_mcc = [0] * NUM_CLASSES
#         test_total_tn_mcc = [0] * NUM_CLASSES
#         test_total_fp_mcc = [0] * NUM_CLASSES
#         test_total_fn_mcc = [0] * NUM_CLASSES

#         test_total_ditch_predicted_as_stream = 0
#         test_total_ditch_predicted_as_background = 0
#         test_total_predicted_ditches = 0

#         with torch.no_grad():
#             for i, (images, labels) in enumerate(test_loader):
#                 outputs_cropped = test_model.predict(images)
#                 squeezed_labels = labels.squeeze(1).long().to(device) 

#                 loss = criterion(outputs_cropped, squeezed_labels)
#                 test_loss += loss.item() * images.size(0)

#                 preds = torch.argmax(outputs_cropped, dim=1)
#                 preds_flat = preds.view(-1)
#                 labels_flat = squeezed_labels.view(-1)


#                 if (i + 1) % 10 == 0 or (i+1) == len(test_loader):
#                     print(f"   Processed test batch [{i+1}/{len(test_loader)}]", flush=True)

#                 batch_intersections, batch_unions = calculate_iou(outputs_cropped, squeezed_labels, NUM_CLASSES)
#                 batch_tp_recall, batch_ap_recall = calculate_recall(outputs_cropped, squeezed_labels, NUM_CLASSES)
#                 batch_tp_f1_test, batch_fp_f1_test, batch_fn_f1_test = calculate_f1_components(outputs_cropped, squeezed_labels, NUM_CLASSES)
#                 batch_tp_mcc_test, batch_tn_mcc_test, batch_fp_mcc_test, batch_fn_mcc_test = calculate_mcc_components(outputs_cropped, squeezed_labels, NUM_CLASSES)

#                 for cls in range(NUM_CLASSES):
#                     test_total_intersections[cls] += batch_intersections[cls]
#                     test_total_unions[cls] += batch_unions[cls]
#                     test_total_true_positives_recall[cls] += batch_tp_recall[cls]
#                     test_total_actual_positives_in_label[cls] += batch_ap_recall[cls]
#                     test_total_tp_f1[cls] += batch_tp_f1_test[cls]
#                     test_total_fp_f1[cls] += batch_fp_f1_test[cls]
#                     test_total_fn_f1[cls] += batch_fn_f1_test[cls]
#                     test_total_tp_mcc[cls] += batch_tp_mcc_test[cls]
#                     test_total_tn_mcc[cls] += batch_tn_mcc_test[cls] 
#                     test_total_fp_mcc[cls] += batch_fp_mcc_test[cls] 
#                     test_total_fn_mcc[cls] += batch_fn_mcc_test[cls] 


#                 ditch_pred_stream_actual_mask = (preds_flat == 1) & (labels_flat == 2)
#                 ditch_pred_background_actual_mask = (preds_flat == 1) & (labels_flat == 0)
#                 test_total_ditch_predicted_as_stream += ditch_pred_stream_actual_mask.sum().item()
#                 test_total_ditch_predicted_as_background += ditch_pred_background_actual_mask.sum().item()
#                 test_total_predicted_ditches += (preds_flat == 1).sum().item()


#         test_loss /= len(test_dataset)
#         test_avg_class_iou = []
#         test_avg_class_recall = []
#         test_avg_class_f1 = []
#         test_avg_class_mcc = []
#         test_fraction_stream_as_ditch = test_total_ditch_predicted_as_stream / test_total_predicted_ditches if test_total_predicted_ditches > 0 else 0.0
#         test_fraction_background_as_ditch = test_total_ditch_predicted_as_background / test_total_predicted_ditches if test_total_predicted_ditches > 0 else 0.0

#         print("\n--- Test Results ---")
#         print(f"Test Loss: {test_loss:.4f}")

#         for cls in range(NUM_CLASSES):
#             iou = test_total_intersections[cls] / test_total_unions[cls] if test_total_unions[cls] > 0 else 0.0
#             test_avg_class_iou.append(iou)

#             recall_test = test_total_true_positives_recall[cls] / test_total_actual_positives_in_label[cls] if test_total_actual_positives_in_label[cls] > 0 else 0.0
#             test_avg_class_recall.append(recall_test)

#             tp_f1, fp_f1, fn_f1 = test_total_tp_f1[cls], test_total_fp_f1[cls], test_total_fn_f1[cls]
#             precision_f1 = tp_f1 / (tp_f1 + fp_f1) if (tp_f1 + fp_f1) > 0 else 0.0
#             recall_f1_metric = tp_f1 / (tp_f1 + fn_f1) if (tp_f1 + fn_f1) > 0 else 0.0 
#             f1_score = 2 * (precision_f1 * recall_f1_metric) / (precision_f1 + recall_f1_metric) if (precision_f1 + recall_f1_metric) > 0 else 0.0
#             test_avg_class_f1.append(f1_score)

#             tp, tn, fp, fn = test_total_tp_mcc[cls], test_total_tn_mcc[cls], test_total_fp_mcc[cls], test_total_fn_mcc[cls]
#             numerator = (tp * tn) - (fp * fn)
#             denominator_val_sqrt_term = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
#             denominator_val_sqrt = math.sqrt(denominator_val_sqrt_term) if denominator_val_sqrt_term > 0 else 0
#             mcc = 0.0 if denominator_val_sqrt == 0 else numerator / denominator_val_sqrt
#             test_avg_class_mcc.append(mcc)


#         metric_names_test = ["IoU", "Recall", "F1-Score", "MCC"]
#         all_metrics_test = [test_avg_class_iou, test_avg_class_recall, test_avg_class_f1, test_avg_class_mcc]

#         print(f"Test Fraction Pred Ditch Actual Stream: {test_fraction_stream_as_ditch:.4f}")
#         print(f"Test Fraction Pred Ditch Actual Background: {test_fraction_background_as_ditch:.4f}")

#         header_test = "Class | " + " | ".join(metric_names_test)
#         print(header_test)
#         print("-" * len(header_test))
#         for cls_idx in range(NUM_CLASSES):
#             metrics_str = f"  {cls_idx}  | " + " | ".join([f"{all_metrics_test[j][cls_idx]:.4f}" for j in range(len(metric_names_test))])
#             print(metrics_str)

#         print("-" * len(header_test))
#         mean_iou_test = np.mean(test_avg_class_iou)
#         mean_recall_test = np.mean(test_avg_class_recall)
#         mean_f1_test = np.mean(test_avg_class_f1)
#         mean_mcc_test = np.mean(test_avg_class_mcc)


#         print(f"Mean  | {mean_iou_test:.4f} | {mean_recall_test:.4f} | {mean_f1_test:.4f} | {mean_mcc_test:.4f}")

#         with open(f'{logdir}/training.log', 'a') as log_file:
#             log_file.write("\n--- Test Results ---\n")
#             log_file.write(f"Test Loss: {test_loss:.4f}\n")
#             log_file.write(f"Test Fraction Pred Ditch Actual Stream: {test_fraction_stream_as_ditch:.4f}\n")
#             log_file.write(f"Test Fraction Pred Ditch Actual Background: {test_fraction_background_as_ditch:.4f}\n")
#             log_file.write(header_test + "\n")
#             log_file.write("-" * len(header_test) + "\n")
#             for cls_idx in range(NUM_CLASSES):
#                  metrics_str = f"  {cls_idx}  | " + " | ".join([f"{all_metrics_test[j][cls_idx]:.4f}" for j in range(len(metric_names_test))])
#                  log_file.write(metrics_str + "\n")
#             log_file.write("-" * len(header_test) + "\n")
#             log_file.write(f"Mean  | {mean_iou_test:.4f} | {mean_recall_test:.4f} | {mean_f1_test:.4f} | {mean_mcc_test:.4f}\n")

#     else:
#         print(f"Error: Best model checkpoint not found at {best_model_path}. Skipping test phase.", flush=True)