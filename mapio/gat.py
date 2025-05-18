import os
import warnings
import tifffile
import random 
import numpy as np
import pandas as pd
from tqdm import tqdm
import graph_processing
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from datetime import datetime

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 8)
pd.set_option("display.max_rows", 10)

NUMPY_SEED = 42
TF_SEED = 42
PYTHON_RANDOM_SEED = 42

def calculate_angle(x_vec, y_vec):
    angle = np.arctan2(y_vec, x_vec) * (25000 / np.pi)
    return angle

def join_graph_dfs(graph_dfs):
    node_dfs = []
    edge_dfs = []
    start_index = 0
    
    for i, (node_df, edge_df) in enumerate(graph_dfs):
        node_df = node_df.reset_index(drop=True)
        person_lookup = node_df["node_id"].reset_index().set_index("node_id") + start_index

        if not person_lookup.empty:
            start_index = person_lookup['index'].max() + 1

        node_df["graph_id"] = i
        node_df["node_id"] = person_lookup.loc[node_df["node_id"]].reset_index()["index"]

        edge_df["target"] = person_lookup.reindex(edge_df["target"]).reset_index()["index"]
        edge_df["source"] = person_lookup.reindex(edge_df["source"]).reset_index()["index"]

        edge_df = edge_df.dropna().astype(int)

        node_dfs.append(node_df)
        edge_dfs.append(edge_df)

    nodes_df = pd.concat(node_dfs).reset_index(drop=True)
    edges_df = pd.concat(edge_dfs).reset_index(drop=True)
    return nodes_df, edges_df

def split_on_file(combined, files):
    nodes_list = []
    files_in_combined = np.unique(files)
    for file_id in files_in_combined:
        file_mask = files == file_id
        nodes = combined[file_mask]
        nodes_list.append(nodes)
    return files_in_combined, nodes_list

def batch_node_data(node_data, edge_data, BATCH_SIZE):
    n_graphs = len(node_data)
    n_batches = n_graphs // BATCH_SIZE
    joined_node_data = []
    joined_edge_data = []
    for i in range(n_batches):
        if i == n_batches-1:
            node_dfs = node_data[i*BATCH_SIZE:]
            edge_dfs = edge_data[i*BATCH_SIZE:]
        else:
            node_dfs = node_data[i*BATCH_SIZE: (i*BATCH_SIZE) + BATCH_SIZE]
            edge_dfs = edge_data[i*BATCH_SIZE: (i*BATCH_SIZE) + BATCH_SIZE]

        joined_nodes, joined_edges = join_graph_dfs(zip(node_dfs, edge_dfs))
        joined_node_data.append(joined_nodes)
        joined_edge_data.append(joined_edges)
    
    return joined_node_data, joined_edge_data

class PrecisionCSVLogger(tf.keras.callbacks.CSVLogger):
    def __init__(self, filename, precision=4, **kwargs):
        super().__init__(filename, **kwargs)
        self.precision = precision

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Format floats to desired precision
        formatted_logs = {
            k: f"{v:.{self.precision}f}" if isinstance(v, (float, np.float32, np.float64)) else v
            for k, v in logs.items()
        }
        super().on_epoch_end(epoch, formatted_logs)

def main(log_dir, epochs):
    data_dir = './logs/m1/test_gat_pca4/graphs'

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
    node_names = ["node_id", "center_x", "center_y", "prob_0", "prob_1", "prob_2", *deep_feats, "target"]
    edge_names = ["target", "source"]

    # --- Load TRAIN/TEST Data ---
    all_node_data = []
    all_edge_data = []
    for graph_name in combined_filenames:
        node_path = os.path.join(data_dir, f'{graph_name[0]}')
        edge_path = os.path.join(data_dir, f'{graph_name[1]}')
        node_df = pd.read_csv(node_path, header=None, sep=',', names=node_names, na_values='_')
        edge_df = pd.read_csv(edge_path, header=None, sep=',', names=edge_names)
        node_df["file_name"] = graph_name[2]
        # Make graph undirected
        edge_df = pd.concat([edge_df, edge_df.rename(columns={"source": "target", "target": "source"})])

        if edge_df.empty: # dont append edges to node list that contain no edges
            continue

        all_node_data.append(node_df)
        all_edge_data.append(edge_df)
    
    BATCH_SIZE = 4
    all_node_data, all_edge_data = batch_node_data(all_node_data, all_edge_data, BATCH_SIZE)

    TRAIN_SPLIT = 0.70
    VALIDATION_SPLIT = 0.10

    # Split the data into training, validation, and test sets
    train_node_data = []
    train_edge_data = []
    test_node_data = []
    test_edge_data = []

    np.random.seed(NUMPY_SEED)
    tf.random.set_seed(TF_SEED)
    random.seed(PYTHON_RANDOM_SEED)

    # get random train indices
    num_train_graphs = int(len(all_node_data) * TRAIN_SPLIT)
    random_train_indices = random.sample(range(len(all_node_data)), num_train_graphs)
    for i, (node_df, edge_df) in enumerate(zip(all_node_data, all_edge_data)):
        # Split the data into training and test sets
        if i in random_train_indices:
            train_node_data.append(node_df)
            train_edge_data.append(edge_df)
        else:
            test_node_data.append(node_df)
            test_edge_data.append(edge_df)

    num_validation_graphs = round(len(train_node_data) * VALIDATION_SPLIT)
    random_validation_indices = random.sample(range(len(train_node_data)), num_validation_graphs)
    
    validation_node_data = [train_node_data[i] for i in random_validation_indices]
    validation_edge_data = [train_edge_data[i] for i in random_validation_indices]

    train_node_data = [graph for i, graph in enumerate(train_node_data) if i not in random_validation_indices]
    train_edge_data = [graph for i, graph in enumerate(train_edge_data) if i not in random_validation_indices]

    # --- End LIST - TRAIN/TEST Data Loading ---

    # Preprocessing data for GAT model - TRAIN - TEST (TENSORS)
    def process_graphs(node_data, edge_data):
        processed_graphs = []
        for (node_df, edge_df) in list(zip(node_data, edge_data)):
            
            node_features = node_df[["center_x", "center_y", "prob_0", "prob_1", "prob_2"]].values.astype(np.float32) # , *deep_feats
            targets = pd.Categorical(node_df["target"], categories=[0,1,2])
            targets = pd.get_dummies(targets)
            graph_ids = node_df["graph_id"].values.astype(np.int32)
            file_name = node_df["file_name"]

            processed_graphs.append(((node_features, edge_df, graph_ids, file_name), targets))
        return processed_graphs

    processed_train_graphs = process_graphs(train_node_data, train_edge_data)
    processed_validation_graphs = process_graphs(validation_node_data, validation_edge_data)
    processed_test_graphs = process_graphs(test_node_data, test_edge_data)

    min_coord = 0
    max_coord = 500
    coord_range = max_coord - min_coord

    epsilon = 1e-7
    if coord_range < epsilon:
        coord_range = epsilon

    def normalize_coords(data):
        return (data - min_coord) / coord_range
    
    def normalize_graphs(graphs):
        for graph in graphs:
            print(graph[0][0][:,1:3])
            print(normalize_coords(graph[0][0][:,1:3]))
            graph[0][0][:,1:3] = normalize_coords(graph[0][0][:,1:3])
        return graphs
    
    processed_train_graphs = normalize_graphs(processed_train_graphs)
    processed_validation_graphs = normalize_graphs(processed_validation_graphs)
    processed_test_graphs = normalize_graphs(processed_test_graphs)

    print("\n--- Graphs Processed ---")
    print(f"Train graphs: {len(processed_train_graphs)}")
    print(f"Validation graphs: {len(processed_validation_graphs)}")
    print(f"Test graphs: {len(processed_test_graphs)}")
    # --- END - Preprocessing data for GAT model - TRAIN - TEST (TENSORS) ---

    def create_tf_dataset(graphs):
        """
            Convert processed graphs into a TensorFlow dataset.
            Each graph is treated as a training instance.
        """
        def generator():
            for graph in graphs:
                yield graph

        output_signature = ((tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
                             tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
                             tf.TensorSpec(shape=(None,), dtype=tf.int32),
                             tf.TensorSpec(shape=(None,), dtype=tf.int32)
                            ),
                            tf.TensorSpec(shape=(None, 3), dtype=tf.int32))

        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train_dataset = create_tf_dataset(processed_train_graphs)
    validation_dataset = create_tf_dataset(processed_validation_graphs)
    test_dataset = create_tf_dataset(processed_test_graphs)

    # # Example: Inspect the first batch in the dataset
    # for batch in train_dataset.take(1):
    #     print("\nSample TensorFlow Train dataset batch:")
    #     print(f"Node features shape: {batch['node_features'].shape}")
    #     print(f"Node features shape: {batch['edge_index'].shape}")
    #     print(f"Targets shape: {batch['targets'].shape}")

    # for batch in validation_dataset.take(1):
    #     print("\nSample TensorFlow Validation dataset batch:")
    #     print(f"Node features shape: {batch['node_features'].shape}")
    #     print(f"Node features shape: {batch['edge_index'].shape}")
    #     print(f"Targets shape: {batch['targets'].shape}")

    # for batch in test_dataset.take(1):
    #     print("\nSample TensorFlow Test dataset batch:")
    #     print(f"Node features shape: {batch['node_features'].shape}")
    #     print(f"Edge index shape: {batch['edge_index'].shape}")
    #     print(f"Targets shape: {batch['targets'].shape}")

    # print("\nTensorFlow dataset created and ready for GAT model training.")


    class GraphAttention(layers.Layer):
        def __init__(
            self,
            units,
            attention="std",
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.units = units
            self.dense1 = layers.Dense(units, activation="relu")
            self.dense2 = layers.Dense(units, activation=None)
            self.kernel_initializer = keras.initializers.get(kernel_initializer)
            self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

            attention_funcs = {
                "std": (self.standard_attention, self.standard_build),
                "cos": (self.cosine_attention, self.cosine_build)
            }
            try:
                self.attention = attention_funcs[attention][0]
                self.build = attention_funcs[attention][1]
            except KeyError:
                print("Invalid attention function string, possible options are ['std', 'cos'], defaulting to std")
                self.attention = attention_funcs["std"][0]
                self.build = attention_funcs["std"][1]

        def standard_build(self, input_shape):
            self.kernel_attention = self.add_weight(
                shape=(self.units * 2, 1),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                name="kernel_attention",
            )
            self.built = True
        
        def cosine_build(self, input_shape):
            self.built = True

        def call(self, inputs):
            node_states, edges = inputs

            # if tf.reduce_any(edges < 0):
                # raise ValueError("--- THERE ARE SOME NEGATIVE VALUED EDGES ---")

            x = self.dense1(node_states)
            node_states_transformed = self.dense2(x)

            attention_scores = self.attention(node_states_transformed, edges)
            attention_scores_norm = self.attention_norm(attention_scores, edges)

            # (3) Gather node states of neighbors, apply attention scores and aggregate
            node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
            out = tf.math.unsorted_segment_sum(
                data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
                segment_ids=edges[:, 0],
                num_segments=tf.shape(node_states)[0],
            )

            return out
        
        def cosine_attention(self, node_states_transformed, edges):
            source_transformed = tf.gather(node_states_transformed, edges[:, 1])
            target_transformed = tf.gather(node_states_transformed, edges[:, 0])

            dot_product = tf.reduce_sum(source_transformed * target_transformed, axis=1)
            source_norm = tf.norm(source_transformed, axis=1)
            target_norm = tf.norm(target_transformed, axis=1)

            attention_scores = dot_product / (source_norm * target_norm + 1e-8)
            return attention_scores
        
        def standard_attention(self, node_states_transformed, edges):
            node_states_expanded = tf.gather(node_states_transformed, edges)
            node_states_expanded = tf.reshape(
                node_states_expanded, (tf.shape(edges)[0], -1)
            )
            attention_scores = tf.nn.leaky_relu(
                tf.matmul(node_states_expanded, self.kernel_attention)
            )
            attention_scores = tf.squeeze(attention_scores, -1)
            return attention_scores
        
        def attention_norm(self, attention_scores, edges):
            attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
            attention_scores_sum = tf.math.unsorted_segment_sum(
                data=attention_scores,
                segment_ids=edges[:, 0],
                num_segments=tf.reduce_max(edges[:, 0]) + 1,
            )
            attention_scores_sum = tf.repeat(
                attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
            )

            attention_scores_norm = attention_scores / attention_scores_sum
            return attention_scores_norm

    class MultiHeadGraphAttention(layers.Layer):
        def __init__(self, units, num_heads=8, attention="std", merge_type="concat", **kwargs):
            super().__init__(**kwargs)
            self.num_heads = num_heads
            self.merge_type = merge_type
            self.attention_layers = [GraphAttention(units, attention) for _ in range(num_heads)]

        def call(self, inputs):
            atom_features, pair_indices = inputs

            # Obtain outputs from each attention head
            outputs = [
                attention_layer([atom_features, pair_indices])
                for attention_layer in self.attention_layers
            ]
            # Concatenate or average the node states from each head
            if self.merge_type == "concat":
                outputs = tf.concat(outputs, axis=-1)
            else:
                outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
            # Activate and return node states
            return tf.nn.relu(outputs)


    class GraphAttentionNetwork(keras.Model):
        def __init__(
            self,
            hidden_units,
            num_heads,
            num_layers,
            output_dim,
            attention="std",
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
            self.attention_layers = [
                MultiHeadGraphAttention(hidden_units, num_heads, attention) for _ in range(num_layers)
            ]
            self.output_layer = layers.Dense(output_dim)

        def call(self, inputs):
            node_states, edges = inputs
            x = self.preprocess(node_states)
            for attention_layer in self.attention_layers:
                x = attention_layer([x, edges]) + x
            outputs = self.output_layer(x)

            return tf.nn.softmax(outputs)

        def train_step(self, data):
            (features_data, edges_data, *_), targets = data

            with tf.GradientTape() as tape:
                # Forward pass
                outputs = self([features_data, edges_data])
                # Compute loss
                loss = self.compiled_loss(targets, outputs)
            # Compute gradients
            grads = tape.gradient(loss, self.trainable_weights)
            # Apply gradients (update weights)
            optimizer.apply_gradients(zip(grads, self.trainable_weights))
            # Update metric(s)
            self.compiled_metrics.update_state(targets, outputs)

            return {m.name: m.result() for m in self.metrics}

        def predict_step(self, data):
            (features_data, edges_data, *_), _ = data
            # Forward pass
            outputs = self([features_data, edges_data])
            # Compute probabilities
            return outputs

        def test_step(self, data):
            (features_data, edges_data, *_), targets = data
            # Forward pass
            outputs = self([features_data, edges_data], training=False)
            # Compute loss
            loss = self.compiled_loss(targets, outputs)
            # Update metric(s)
            self.compiled_metrics.update_state(targets, outputs)
            return {m.name: m.result() for m in self.metrics}

    # --- Hyperparameter Search Setup ---
    hidden_units = 100
    num_heads = 11

    # Fixed parameters
    NUM_LAYERS = 5
    OUTPUT_DIM = 3
    LEARNING_RATE = 1e-3
    ATTENTION_TYPE = "std"

    print("\n--- Starting Hyperparameter Search ---")
    # print(f"HIDDEN_UNITS options: {hidden_units_list}")
    # print(f"NUM_HEADS options: {num_heads_list}")
    print(f"Fixed NUM_LAYERS: {NUM_LAYERS}")
    print(f"Fixed LEARNING_RATE: {LEARNING_RATE}")
    print(f"Training Epochs per trial: {epochs}")

    best_val_loss = float('inf')
    best_params = {}
    all_results = []

    # --- Loop through Hyperparameters ---
    
    current_params = {'hidden_units': hidden_units, 'num_heads': num_heads}
    print(f"\n--- Running Trial: {current_params} ---")
    timestamp = datetime.now()
    trial_log_dir = os.path.join(log_dir, "gat", timestamp.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(trial_log_dir, exist_ok=True)
    print(f"Logging trial results to: {trial_log_dir}")

    loss_fn = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, weight_decay=0.01)
    metrics = [keras.metrics.IoU(3, [0,1,2]),
               keras.metrics.Recall(name="rec"),
               keras.metrics.Recall(class_id=0, name='rec_0'),
               keras.metrics.Recall(class_id=1, name='rec_1'),
               keras.metrics.Recall(class_id=2, name='rec_2'),
               keras.metrics.Precision(name="prec"),
               keras.metrics.Precision(class_id=0, name='prec_0'),
               keras.metrics.Precision(class_id=1, name='prec_1'),
               keras.metrics.Precision(class_id=2, name='prec_2')] 
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(trial_log_dir, "model.h5"), save_weights_only=True, monitor="val_loss")
    checkpoint_best = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(trial_log_dir, "model_best.h5"), save_weights_only=True, monitor="val_loss", save_best_only=True)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=1e-5,
        restore_best_weights=True, 
    )
    csv_logger = PrecisionCSVLogger(os.path.join(trial_log_dir, 'training.log'), precision=4)
    callbacks = [csv_logger, checkpoint, checkpoint_best] 

    np.random.seed(NUMPY_SEED)
    tf.random.set_seed(TF_SEED)
    random.seed(PYTHON_RANDOM_SEED)

    gat_model = GraphAttentionNetwork(
        hidden_units=hidden_units,
        num_heads=num_heads,
        num_layers=NUM_LAYERS,
        output_dim=OUTPUT_DIM,
        attention=ATTENTION_TYPE
    )

    # Compile model
    gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)

    # class_weight = {0: 1, 1: 1, 2: 1}
    # Train the model
    history = gat_model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        # class_weight=class_weight
    )

    print("\n--- Model Training Complete for Trial ---")
    for metric in metrics:
        metric.reset_state()

    for (node_features, edge_df, graph_ids, file_name), targets in tqdm(test_dataset):
        predictions = gat_model((node_features, edge_df))

        files_in_combined, predictions_list = split_on_file(predictions, file_name)
        for file_id, graph_preds in zip(files_in_combined, predictions_list):
            rec_data_dir = os.path.join("./logs/m1/test_gat_pca4/reconstruction", f"{file_id}.npz")
            rec_data = np.load(rec_data_dir)
            node_mask = rec_data["image"]
            unet_pred = rec_data["unet_pred"]
            image_name = rec_data["image_name"]

            gt_path = os.path.join("./data/05m_chips/labels/", f"{image_name}.tif")
            gt = tifffile.imread(gt_path)

            gat_pred = graph_processing.graph_to_image(np.argmax(graph_preds.numpy(), axis=1), node_mask)

            for metric in metrics:
                metric.update_state(gt, gat_pred)
    
    with open(os.path.join(trial_log_dir, "test.csv"), "w") as f:
        names = [metric.name for metric in metrics]

        results = [metric.result().numpy() for metric in metrics]
        result_strings = [f"{r:.4f}" for r in results]

        f.write(f'{",".join(names)}\n')
        f.write(f'{",".join(result_strings)}\n')          

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Tensorflow model for Osaka Shopping Mall',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('log_dir', help='Folder to write log data to')
    parser.add_argument('--epochs', help='Number of epochs to train', type=int,
                        default=10)

    args = vars(parser.parse_args())
    main(**args)