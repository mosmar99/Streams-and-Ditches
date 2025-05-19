import os
import pandas as pd
import torch
import metrics
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool, global_max_pool

def read_graphs(data_dir):
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

    deep_feats = [f"deep_{i}" for i in range(4)]
    slope_feature_names = ["slope_min", "slope_mean", "slope_max", "slope_std", "area"]
    node_names = ["node_id", "center_x", "center_y", "prob_0", "prob_1", "prob_2", *slope_feature_names, *deep_feats, "target"]
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

        # Add self-loops to the edge DataFrame
        node_ids = node_df['node_id'].unique()
        self_loops = pd.DataFrame({
            'source': node_ids,
            'target': node_ids
        })
        edge_df = pd.concat([edge_df, self_loops], ignore_index=True)


        if edge_df.empty: # dont append edges to node list that contain no edges
            continue

        all_node_data.append(node_df)
        all_edge_data.append(edge_df)

    datas = []
    for node_df, edge_df in zip(all_node_data, all_edge_data):
        # Extract edge_index
        edge_index = torch.tensor(edge_df[['source', 'target']].values.T, dtype=torch.long)

        # Extract node features (excluding 'node_id', 'center_x', 'center_y', 'file_name', 'target')
        feature_columns = ["prob_0", "prob_1", "prob_2", *slope_feature_names, *deep_feats]
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
                 num_classes, heads_conv1=2, heads_conv2=1,
                 dropout_rate=0): # Removed pool_type
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        self.conv1 = GATv2Conv(in_channels, hidden_channels_gnn, heads=heads_conv1, concat=True, dropout=dropout_rate)
        self.conv2 = GATv2Conv(hidden_channels_gnn * heads_conv1, out_channels_gnn, heads=heads_conv2, concat=False, dropout=dropout_rate)

        self.node_classifier = torch.nn.Linear(out_channels_gnn, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv2(x, edge_index)
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
            
            loss = criterion(out, data.y) # Compares node logits with node labels
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
            lr_scheduler=None, checkpoint_handlers=None, metrics_handler=None):

        history = {'train_loss': [], 'train_metrics': [], 'val_loss': [], 'val_metrics': []}

        self.to(device)

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current LR: {current_lr:.6f}")

            train_loss, train_metrics = self._train_epoch(train_loader, optimizer, criterion, device, metrics_handler=metrics_handler)
            val_loss, val_metrics = self._validate_epoch(val_loader, criterion, device, metrics_handler=metrics_handler)

            print(f"Train Loss (node-avg): {train_loss:.4f}, Train Acc (node): {train_metrics}")
            print(f"Val Loss (node-avg): {val_loss:.4f}, Val Acc (node): {val_metrics}")

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

    SEED = 42
    BATCH_SIZE = 8
    CLASSES = [0,1,2]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_dir = './logs/m1/test_gat_stats/graphs'
    log_dir = os.path.join('./logs/torch_gat/', timestamp)
    os.makedirs(log_dir, exist_ok=True)

    datas = read_graphs(data_dir)

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

    num_node_features = 12
    gnn_hidden_dim = 128
    gnn_output_dim = 128
    num_graph_classes = 3
    heads_conv1=11
    heads_conv2=11
    learning_rate = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = GATv2Net_NodeClassifier(
        in_channels=num_node_features,
        hidden_channels_gnn=gnn_hidden_dim,
        out_channels_gnn=gnn_output_dim,
        num_classes=num_graph_classes,
        dropout_rate=0.2,
        heads_conv1=heads_conv1,
        heads_conv2=heads_conv2
    )

    checkpoints = [MinCheckpoint(log_dir), Checkpoint(log_dir)]
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    history = model.fit(
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=500, # Small number of epochs for example
        device=device,
        checkpoint_handlers=checkpoints,
        metrics_handler=metrics_handler
    )

    # print("\nTraining History:")
    # for key, values in history.items():
    #     print(f"{key}: {[f'{v:.4f}' for v in values]}")

if __name__ == '__main__':
    main()