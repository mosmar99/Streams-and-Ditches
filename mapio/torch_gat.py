import os
import pandas as pd
import torch
import metrics
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Linear
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GATv2Conv, GATConv, global_mean_pool, global_add_pool, global_max_pool

def read_graphs(data_dir, num_deep_feats):
    def normalize_coords(data, minimum, range):
        return (data - minimum) / range

    files = os.listdir(data_dir)
    node_files = sorted([f for f in files if f.endswith('.nodes')])
    edge_files = sorted([f for f in files if f.endswith('.edges')])

    node_base_names = [f.replace('.nodes', '') for f in node_files]
    edge_base_names = [f.replace('.edges', '') for f in edge_files]

    if len(node_base_names) != len(edge_base_names) or set(node_base_names) != set(edge_base_names):
        raise ValueError("Mismatch or inconsistency between node and edge files found.")

    node_dict = {f.replace('.nodes', ''): f for f in node_files}
    combined_filenames = []
    for edge_file in edge_files:
        base_name = edge_file.replace('.edges', '')
        combined_filenames.append((node_dict[base_name], edge_file, base_name))

    deep_feats = [f"deep_{i}" for i in range(num_deep_feats)]
    slope_feature_names = ["slope_min", "slope_mean", "slope_max", "slope_std", "area"]
    node_names = ["node_id", "center_x", "center_y", "prob_0", "prob_1", "prob_2", *slope_feature_names, *deep_feats, "target"] #, *slope_feature_names
    edge_names = ["target", "source"]

    # --- Load TRAIN/TEST Data ---
    all_node_data = []
    all_edge_data = []
    for graph_name in combined_filenames:
        node_path = os.path.join(data_dir, f'{graph_name[0]}')
        edge_path = os.path.join(data_dir, f'{graph_name[1]}')
        node_df = pd.read_csv(node_path, header=None, sep=',', names=node_names, na_values='_', dtype=np.float32)
        edge_df = pd.read_csv(edge_path, header=None, sep=',', names=edge_names, dtype=np.int32)
        node_df["file_name"] = graph_name[2]
        # Make graph undirected
        edge_df = pd.concat([edge_df, edge_df.rename(columns={"source": "target", "target": "source"})])

        node_df["area"] = normalize_coords(node_df["area"], minimum=10, range=100)

        # # Add self-loops to the edge DataFrame
        # node_ids = node_df['node_id'].unique()
        # self_loops = pd.DataFrame({
        #     'source': node_ids,
        #     'target': node_ids
        # })
        # edge_df = pd.concat([edge_df, self_loops], ignore_index=True)


        if edge_df.empty: # dont append edges to node list that contain no edges
            continue

        all_node_data.append(node_df)
        all_edge_data.append(edge_df)

    datas = []
    for node_df, edge_df in zip(all_node_data, all_edge_data):
        # Extract edge_index
        edge_index = torch.tensor(edge_df[['source', 'target']].values.T, dtype=torch.long)

        # Extract node features (excluding 'node_id', 'center_x', 'center_y', 'file_name', 'target')
        feature_columns = ["prob_0", "prob_1", "prob_2", *slope_feature_names, *deep_feats] # , *slope_feature_names
        x = torch.tensor(node_df[feature_columns].values, dtype=torch.float32)

        # Extract target (optional: for node classification/regression)
        y = torch.tensor(node_df['target'].values, dtype=torch.long)

        # Save original node IDs (optional, useful for mapping back)
        original_node_ids = torch.tensor(node_df['node_id'].values, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.original_node_ids = original_node_ids
        data.graph_id = node_df['file_name'].iloc[0]  # for tracking

        datas.append(data)
    
    return datas

class MinCheckpoint():
    def __init__(self, logdir):
        self.min_loss = np.inf
        self.logdir = logdir

    def save(self, model, loss):
        if loss < self.min_loss:
            print(f" -- Updated Checkpoint: {self.min_loss} > {loss}", flush=True)
            self.min_loss = loss
            torch.save(model.state_dict(), os.path.join(self.logdir, 'gat_model_min_ckpt.pth'))

class Checkpoint():
    def __init__(self, logdir):
        self.logdir = logdir

    def save(self, model, loss):
        torch.save(model.state_dict(), os.path.join(self.logdir, 'gat_model_ckpt.pth'))

class GATv2Net_NodeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_gnn, out_channels_gnn,
                 num_classes, heads=11,
                 dropout_rate=0): # Removed pool_type
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        self.lin1 = Linear(in_channels, in_channels)
        self.conv1 = GATv2Conv(in_channels, hidden_channels_gnn, heads=heads, concat=True, dropout=dropout_rate, residual=True)

        self.lin2 = Linear(hidden_channels_gnn * heads, hidden_channels_gnn * heads)
        self.conv2 = GATv2Conv(hidden_channels_gnn * heads, hidden_channels_gnn, heads=heads, concat=True, dropout=dropout_rate, residual=True)

        self.lin3 = Linear(hidden_channels_gnn * heads, hidden_channels_gnn * heads)
        self.conv3 = GATv2Conv(hidden_channels_gnn * heads, hidden_channels_gnn, heads=heads, concat=True, dropout=dropout_rate, residual=True)

        self.lin4 = Linear(hidden_channels_gnn * heads, hidden_channels_gnn * heads)
        self.conv4 = GATv2Conv(hidden_channels_gnn * heads, out_channels_gnn, heads=heads, concat=False, dropout=dropout_rate, residual=True)

        self.node_classifier = torch.nn.Linear(out_channels_gnn, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.lin2(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.lin3(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.lin4(x)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x_node_logits = self.node_classifier(x)

        return x_node_logits
    
    def _train_epoch(self, train_loader, optimizer, criterion, device, metrics_handler=None):
        self.train()
        total_loss = 0
        total_nodes_processed = 0

        metrics_handler.reset()

        for data in tqdm(train_loader, desc="Training", leave=False, unit="batch"):
            data = data.to(device)
            optimizer.zero_grad()

            if data.y is None:
                raise ValueError("Ground truth node labels 'data.y' not found in the batch.")

            out = self(data.x, data.edge_index, data.batch) # out is [total_nodes_in_batch, num_classes]
            
            loss = criterion(out, data.y) # + 0.01*torch.mean(out**2) # Compares node logits with node labels
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_nodes
            metrics_handler.add(out, data.y)
            total_nodes_processed += data.num_nodes

        avg_loss = total_loss / total_nodes_processed if total_nodes_processed > 0 else 0
        metrics = metrics_handler.value()
        return avg_loss, metrics

    def _validate_epoch(self, val_loader, criterion, device, metrics_handler=None):
        self.eval()
        total_loss = 0
        total_nodes_processed = 0

        metrics_handler.reset()

        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation", leave=False, unit="batch"):
                data = data.to(device)
                if data.y is None:
                    raise ValueError("Ground truth node labels 'data.y' not found in the batch.")

                out = self(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                metrics_handler.add(out, data.y)
                total_loss += loss.item() * data.num_nodes
                total_nodes_processed += data.num_nodes


        avg_loss = total_loss / total_nodes_processed if total_nodes_processed > 0 else 0
        metrics = metrics_handler.value(prefix="VAL_")
        return avg_loss, metrics
    
    def fit(self, train_loader, val_loader, optimizer, criterion, epochs, device,
            lr_scheduler=None, checkpoint_handlers=None, metrics_handler=None, log_dir=None):

        history = {'train_loss': [], 'train_metrics': [], 'val_loss': [], 'val_metrics': []}

        names = []
        if log_dir != None:
            with open(os.path.join(log_dir, "training.log"), 'w') as f:
                names = ["LOSS", "VAL_LOSS"]
                names += metrics_handler.labels()
                names += metrics_handler.labels(prefix="VAL_")
                f.write(f'{",".join(names)}\n')

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current LR: {current_lr:.6f}")

            train_loss, train_metrics = self._train_epoch(train_loader, optimizer, criterion, device, metrics_handler=metrics_handler)
            val_loss, val_metrics = self._validate_epoch(val_loader, criterion, device, metrics_handler=metrics_handler)

            if log_dir != None:
                with open(os.path.join(log_dir, "training.log"), 'a') as f:
                    results = train_metrics
                    results.update(val_metrics)
                    results["LOSS"] = train_loss
                    results["VAL_LOSS"] = val_loss
                    result_strings = [f"{results[name]:.4f}" for name in names]
                    f.write(f'{",".join(result_strings)}\n')   

            train_metrics.keys()
            print(f"Train Loss (node-avg): {train_loss:.4f}")
            print(f"Val Loss (node-avg): {val_loss:.4f}")

            history['train_loss'].append(train_loss)
            history['train_metrics'].append(train_metrics)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)

            if lr_scheduler:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_loss)
                else:
                    lr_scheduler.step()

            if checkpoint_handlers:
                for checkpoint_handler in checkpoint_handlers:
                    checkpoint_handler.save(self, val_loss)
        
        return history

    def predict(self, data_loader, device):
        self.to(device) # Ensure model is on the correct device
        self.eval() # Set model to evaluation mode
        all_preds = []
        with torch.no_grad():
            for data in tqdm(data_loader, desc="Predicting", leave=False, unit="batch"):
                data = data.to(device)
                out = self(data.x, data.edge_index, data.batch)
                preds = out.argmax(dim=1)
                all_preds.append(preds)
        if not all_preds:
            return torch.empty(0, dtype=torch.long) # Handle empty dataloader
        return torch.cat(all_preds)

    def predict_softmax(self, data_loader, device):
        self.to(device) # Ensure model is on the correct device
        self.eval() # Set model to evaluation mode
        all_probs = []
        with torch.no_grad():
            for data in tqdm(data_loader, desc="Predicting Softmax", leave=False, unit="batch"):
                data = data.to(device)
                out = self(data.x, data.edge_index, data.batch)
                probs = F.softmax(out, dim=1)
                all_probs.append(probs)
        if not all_probs:
            return torch.empty(0, self.num_classes) # Handle empty dataloader
        return torch.cat(all_probs)
    
def main():
    from datetime import datetime
    import graph_processing
    import tifffile
    import matplotlib.pyplot as plt

    SEED = 42
    BATCH_SIZE = 8
    CLASSES = [0,1,2]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_dir = './logs/m1/test_gat_more/graphs'
    num_deep_feats = 12
    log_dir = os.path.join('./logs/torch_gat/', timestamp)
    os.makedirs(log_dir, exist_ok=True)

    datas = read_graphs(data_dir, num_deep_feats)

    temp_data, test_data = train_test_split(datas, test_size=0.2, random_state=SEED)
    train_data, val_data = train_test_split(temp_data, test_size=0.1, random_state=SEED)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    metrics_list = [
        metrics.MCC(CLASSES, name="MCC"),
        metrics.Recall(CLASSES, name="REC"),
        metrics.Precision(CLASSES, name="PREC"),
        metrics.IoU(CLASSES, name="IOU")]
    
    metrics_handler = metrics.MetricsList(metrics_list)

    num_node_features = 20
    gnn_hidden_dim = 128
    gnn_output_dim = 128
    num_graph_classes = 3
    heads=11
    learning_rate = 1e-3
    weight_decay = 0.03

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = GATv2Net_NodeClassifier(
        in_channels=num_node_features,
        hidden_channels_gnn=gnn_hidden_dim,
        out_channels_gnn=gnn_output_dim,
        num_classes=num_graph_classes,
        dropout_rate=0.2,
        heads=heads
    ).to(device)


    checkpoints = [MinCheckpoint(log_dir), Checkpoint(log_dir)]
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    class_weights = torch.tensor([1.2,1.0,2.8], device=device)
    criterion = CrossEntropyLoss(weight=class_weights)

    history = model.fit(
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=50, # Small number of epochs for example
        device=device,
        checkpoint_handlers=checkpoints,
        metrics_handler=metrics_handler,
        log_dir=log_dir
    )

    # model.load_state_dict(torch.load("./logs/torch_gat/20250519_154852/gat_model_ckpt.pth", map_location=device))

    model.eval()
    reconstruction_data_base_dir = "./logs/m1/test_gat_more/reconstruction"
    gt_data_base_dir = "./data/05m_chips/labels"

    img_metrics_list = [
        metrics.MCC(CLASSES, name="MCC", from_logits=False),
        metrics.Recall(CLASSES, name="REC", from_logits=False),
        metrics.Precision(CLASSES, name="PREC", from_logits=False),
        metrics.IoU(CLASSES, name="IOU", from_logits=False)]
    
    img_metrics_handler = metrics.MetricsList(img_metrics_list)
    img_metrics_handler.reset()

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating Segmentation from Nodes"):
            batch_data = batch_data.to(device)

            node_predictions_logits = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            node_predicted_classes = torch.argmax(node_predictions_logits, dim=1)
            
            # Ensure graph_id is a list (DataLoader might make it a tuple sometimes)
            graph_ids_in_batch = list(batch_data.graph_id) if isinstance(batch_data.graph_id, (list, tuple)) else [batch_data.graph_id]
            
            current_node_idx = 0
            for i in range(batch_data.num_graphs): # Iterate through graphs in the batch
                num_nodes_in_graph = batch_data.ptr[i+1] - batch_data.ptr[i]
                
                # Get predictions for the current graph
                graph_node_preds_classes_tensor = node_predictions_logits[current_node_idx : current_node_idx + num_nodes_in_graph]
                graph_node_preds_classes_tensor_argmax = node_predicted_classes[current_node_idx : current_node_idx + num_nodes_in_graph]
                
                # Get the file_id (assuming graph_id from Data object is the file_id)
                file_id = graph_ids_in_batch[i] 
                
                current_node_idx += num_nodes_in_graph

                # --- Load reconstruction data and GT for this graph/image ---
                rec_data_path = os.path.join(reconstruction_data_base_dir, f"{file_id}.npz")
                rec_data = np.load(rec_data_path)
                node_mask_np = rec_data["image"]      # Mask to map node predictions to pixels
                image_name = str(rec_data["image_name"]) # Ensure it's a string
                unet_pred = rec_data["unet_pred"] # Ensure it's a string

                gt_path = os.path.join(gt_data_base_dir, f"{image_name}.tif")
                gt_image_np = tifffile.imread(gt_path)

                gat_pred_image_np = graph_processing.graph_to_image(
                    graph_node_preds_classes_tensor_argmax.cpu().numpy(),
                    node_mask_np
                )

                gt_tensor = torch.from_numpy(gt_image_np).long()
                gat_pred_tensor = torch.from_numpy(gat_pred_image_np).long()

                # vmin = 0
                # vmax = 2
                # fig, ax = plt.subplots(1,3)
                # ax[0].imshow(unet_pred, vmin=vmin, vmax=vmax)
                # ax[0].set_title("Unet Prediction")
                # ax[1].imshow(gat_pred_image_np, vmin=vmin, vmax=vmax)
                # ax[1].set_title("Gat Prediction")
                # ax[2].imshow(gt_image_np, vmin=vmin, vmax=vmax)
                # ax[2].set_title("Ground Truth")
                # plt.show()

                gt_tensor_flat = gt_tensor.reshape(-1)
                gat_pred_tensor_flat = gat_pred_tensor.reshape(-1)

                img_metrics_handler.add(gat_pred_tensor_flat, gt_tensor_flat)

    print("\n--- Image Segmentation Evaluation Complete ---")
    results_dict = img_metrics_handler.value(prefix="test_seg_")

    if log_dir:
        output_path = os.path.join(log_dir, "test_segmentation_metrics.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            names = sorted(results_dict.keys())
            results_str = [f"{results_dict[name]:.4f}" for name in names]
            f.write(f'{",".join(names)}\n')
            f.write(f'{",".join(results_str)}\n')
        print(f"Segmentation test metrics saved to {output_path}")

    for name, value in results_dict.items():
        print(f"{name}: {value:.4f}")


    # print("\nTraining History:")
    # for key, values in history.items():
    #     print(f"{key}: {[f'{v:.4f}' for v in values]}")

if __name__ == '__main__':
    main()