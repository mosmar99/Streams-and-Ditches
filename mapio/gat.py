import os
import warnings
import random 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 8)
pd.set_option("display.max_rows", 10)

NUMPY_SEED = 42
TF_SEED = 42
PYTHON_RANDOM_SEED = 42

def calculate_angle(x_vec, y_vec):
    angle = np.arctan2(y_vec, x_vec) * (25000 / np.pi)
    return angle

def main(log_dir, epochs):
    data_dir = './dataset'
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
        combined_filenames.append((node_dict[base_name], edge_file))

    node_names = ["person_id", "current_x", "current_y", "previous_x", "previous_y", "future_x", "future_y"]
    edge_names = ["target", "source"]

    # --- Load TRAIN/TEST Data ---
    all_node_data = []
    all_edge_data = []
    for graph_name in combined_filenames:
        node_path = os.path.join(data_dir, f'{graph_name[0]}')
        edge_path = os.path.join(data_dir, f'{graph_name[1]}')
        node_df = pd.read_csv(node_path, header=None, sep=',', names=node_names, na_values='_')
        edge_df = pd.read_csv(edge_path, header=None, sep=',', names=edge_names)

        # Make graph undirected
        edge_df = pd.concat([edge_df, edge_df.rename(columns={"source": "target", "target": "source"})])

        # Make graph undirected
        node_df["prev_step_x"] = node_df["current_x"] - node_df["previous_x"]
        node_df["prev_step_y"] = node_df["current_y"] - node_df["previous_y"]
        node_df["direction"] = calculate_angle(node_df["prev_step_x"], node_df["prev_step_y"])

        node_df = node_df.dropna() # drop missing values in node data

        edge_df = edge_df[~edge_df.isin([-1]).any(axis=1)] # drop edge links with -1 values
        
        if edge_df.empty: # dont append edges to node list that contain no edges
            continue

        all_node_data.append(node_df)
        all_edge_data.append(edge_df)

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
            person_ids = node_df["person_id"].tolist()
            id_to_idx = {pid: idx for idx, pid in enumerate(person_ids)}

            node_features = node_df[["current_x", "current_y",
                                     "previous_x", "previous_y",
                                     "direction"]].values.astype(np.float32)
            targets = node_df[["future_x", "future_y"]].values.astype(np.float32)
            edge_index = edge_df.replace({"source": id_to_idx, "target": id_to_idx}).to_numpy()
            processed_graphs.append({
                "node_features": node_features,
                "edge_index": edge_index,
                "targets": targets,
                "original_person_ids": person_ids
            })
        return processed_graphs

    processed_train_graphs = process_graphs(train_node_data, train_edge_data)
    processed_validation_graphs = process_graphs(validation_node_data, validation_edge_data)
    processed_test_graphs = process_graphs(test_node_data, test_edge_data)

    overall_min_coord = np.inf
    overall_max_coord = -np.inf
    found_data = False
    for graph in processed_train_graphs:
        if graph['node_features'].size > 0:
            min_features = np.min(graph['node_features'])
            max_features = np.max(graph['node_features'])
            overall_min_coord = min(overall_min_coord, min_features)
            overall_max_coord = max(overall_max_coord, max_features)
            found_data = True

        if graph['targets'].size > 0:
            valid_targets_mask = ~np.isnan(graph['targets'])
            if np.any(valid_targets_mask):
                valid_target_values = graph['targets'][valid_targets_mask]
                min_targets = np.min(valid_target_values)
                max_targets = np.max(valid_target_values)
                overall_min_coord = min(overall_min_coord, min_targets)
                overall_max_coord = max(overall_max_coord, max_targets)
                found_data = True

    min_coord = overall_min_coord
    max_coord = overall_max_coord
    coord_range = max_coord - min_coord

    epsilon = 1e-7
    if coord_range < epsilon:
        coord_range = epsilon

    def normalize_coords(data):
        return (data - min_coord) / coord_range
    
    def normalize_graphs(graphs):
        for graph in graphs:
            graph['node_features'] = normalize_coords(graph['node_features'])
            valid_targets_mask = ~np.isnan(graph['targets'])
            graph['targets'][valid_targets_mask] = normalize_coords(graph['targets'][valid_targets_mask])
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
                yield {
                    "node_features": graph["node_features"],
                    "edge_index": graph["edge_index"],
                    "targets": graph["targets"]  
                }

        output_signature = {
            "node_features": tf.TensorSpec(shape=(None, 5), dtype=tf.float32),  
            "edge_index": tf.TensorSpec(shape=(None, 2), dtype=tf.int32),       
            "targets": tf.TensorSpec(shape=(None, 2), dtype=tf.float32)         
        }

        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        return dataset

    train_dataset = create_tf_dataset(processed_train_graphs)
    validation_dataset = create_tf_dataset(processed_validation_graphs)
    test_dataset = create_tf_dataset(processed_test_graphs)

    # Example: Inspect the first batch in the dataset
    for batch in train_dataset.take(1):
        print("\nSample TensorFlow Train dataset batch:")
        print(f"Node features shape: {batch['node_features'].shape}")
        print(f"Node features shape: {batch['edge_index'].shape}")
        print(f"Targets shape: {batch['targets'].shape}")

    for batch in validation_dataset.take(1):
        print("\nSample TensorFlow Validation dataset batch:")
        print(f"Node features shape: {batch['node_features'].shape}")
        print(f"Node features shape: {batch['edge_index'].shape}")
        print(f"Targets shape: {batch['targets'].shape}")

    for batch in test_dataset.take(1):
        print("\nSample TensorFlow Test dataset batch:")
        print(f"Node features shape: {batch['node_features'].shape}")
        print(f"Edge index shape: {batch['edge_index'].shape}")
        print(f"Targets shape: {batch['targets'].shape}")

    print("\nTensorFlow dataset created and ready for GAT model training.")


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
            outputs = node_states[:,:2]
            return outputs

        def train_step(self, data):
            (features_data, edges_data), targets = data

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
            (features_data, edges_data), _ = data
            # Forward pass
            outputs = self([features_data, edges_data])
            # Compute probabilities
            return outputs

        def test_step(self, data):
            (features_data, edges_data), targets = data
            # Forward pass
            outputs = self([features_data, edges_data], training=False)
            # Compute loss
            loss = self.compiled_loss(targets, outputs)
            # Update metric(s)
            self.compiled_metrics.update_state(targets, outputs)
            return {m.name: m.result() for m in self.metrics}

    # --- Hyperparameter Search Setup ---
    hidden_units_list = [100]
    num_heads_list = [11, 8, 5]

    # Fixed parameters
    NUM_LAYERS = 1
    OUTPUT_DIM = 2
    LEARNING_RATE = 1e-5
    ATTENTION_TYPE = "std"

    print("\n--- Starting Hyperparameter Search ---")
    print(f"HIDDEN_UNITS options: {hidden_units_list}")
    print(f"NUM_HEADS options: {num_heads_list}")
    print(f"Fixed NUM_LAYERS: {NUM_LAYERS}")
    print(f"Fixed LEARNING_RATE: {LEARNING_RATE}")
    print(f"Training Epochs per trial: {epochs}")

    best_val_loss = float('inf')
    best_params = {}
    all_results = []

    # --- Loop through Hyperparameters ---
    for hu in hidden_units_list:
        for nh in num_heads_list:
            current_params = {'hidden_units': hu, 'num_heads': nh}
            print(f"\n--- Running Trial: {current_params} ---")

            trial_log_dir = os.path.join(log_dir, f"hu_{hu}_nh_{nh}")
            os.makedirs(trial_log_dir, exist_ok=True)
            print(f"Logging trial results to: {trial_log_dir}")

            loss_fn = keras.losses.MeanSquaredError()
            optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            metrics = [keras.metrics.RootMeanSquaredError(name="rmse"), 
                       keras.metrics.MeanAbsoluteError(name="mae")] 

            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                min_delta=1e-5,
                restore_best_weights=True, 
            )
            csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(trial_log_dir, 'training.log'))
            callbacks = [csv_logger] 

            np.random.seed(NUMPY_SEED)
            tf.random.set_seed(TF_SEED)
            random.seed(PYTHON_RANDOM_SEED)

            gat_model = GraphAttentionNetwork(
                hidden_units=hu,
                num_heads=nh,
                num_layers=NUM_LAYERS,
                output_dim=OUTPUT_DIM,
                attention=ATTENTION_TYPE
            )

            # Compile model
            gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)

            # Train the model
            history = gat_model.fit(
                train_dataset.map(lambda x: (( x["node_features"], x["edge_index"] ), x['targets']) ),
                validation_data=validation_dataset.map(lambda x: (( x["node_features"], x["edge_index"] ), x['targets']) ),
                epochs=epochs,
                callbacks=callbacks,
                verbose=2,
            )

            print("\n--- Model Training Complete for Trial ---")

            print("Evaluating best weights (from this trial) on validation data...")
            val_results = gat_model.evaluate(
                validation_dataset.map(lambda x: (( x["node_features"], x["edge_index"] ), x['targets']) ),
                verbose=2
            )
            current_val_loss = val_results[0]
            current_val_rmse_norm = val_results[1] 
            current_val_mae_norm = val_results[2] 
            current_val_rmse_mm = current_val_rmse_norm * coord_range
            current_val_mae_mm = current_val_mae_norm * coord_range

            print(f"Best Validation Loss for this trial: {current_val_loss:.6f}")
            print(f"Best Validation RMSE (normalized) for this trial: {current_val_rmse_norm:.6f}")
            print(f"Best Validation MAE (normalized) for this trial: {current_val_mae_mm:.6f}")
            print(f"Best Validation RMSE (millimeters) for this trial: {current_val_rmse_mm:.2f} mm") 
            print(f"Best Validation MAE (millimeters) for this trial: {current_val_mae_mm:.2f} mm") 

            print("Evaluating best weights (from this trial) on test data...")
            test_loss, test_rmse_norm, test_mae_norm = gat_model.evaluate(
                test_dataset.map(lambda x: (( x["node_features"], x["edge_index"] ), x['targets']) ),
                verbose=1,
            )

            test_rmse_mm = test_rmse_norm * coord_range
            test_mae_mm = test_mae_norm * coord_range

            print(f"\nTest Loss: {test_loss:.6f}")
            print(f"Test RMSE (normalized): {test_rmse_norm:.6f}") 
            print(f"Test MAE (normalized): {test_mae_norm:.6f}") 
            print(f"Test RMSE (millimeters): {test_rmse_mm:.2f} mm") 
            print(f"Test MAE (millimeters): {test_mae_mm:.2f} mm") 

            trial_results_data = {
                'hidden_units': hu,
                'num_heads': nh,
                'val_loss': current_val_loss,        
                'val_rmse_norm': current_val_rmse_norm, 
                'val_rmse_mm': current_val_rmse_mm,     
                'val_mae_norm': current_val_mae_norm, 
                'val_mae_mm': current_val_mae_mm,     
                'test_loss': test_loss,              
                'test_rmse_norm': test_rmse_norm,     
                'test_rmse_mm': test_rmse_mm,          
                'test_mae_norm': test_mae_norm,     
                'test_mae_mm': test_mae_mm,          
            }
            all_results.append(trial_results_data)

            df_trial = pd.DataFrame([trial_results_data])
            df_trial.to_csv(os.path.join(trial_log_dir, 'metrics.log'), index=False)

            if current_val_loss < best_val_loss:
                print(f"*** New best validation loss found: {current_val_loss:.6f} (improvement from {best_val_loss:.6f}) ***")
                best_val_loss = current_val_loss
                best_params = current_params

    # --- End of Hyperparameter Loop ---

    print("\n--- Hyperparameter Search Complete ---")

    all_results_df = pd.DataFrame(all_results)
    all_results_df.sort_values(by='val_loss', inplace=True)
    summary_path = os.path.join(log_dir, 'hyperparameter_summary.log')
    all_results_df.to_csv(summary_path, index=False)

    print(f"\nSummary of all trials saved to: {summary_path}")
    print("\nBest parameters found based on validation loss:")
    print(best_params)
    print(f"Best validation loss achieved: {best_val_loss:.6f}")

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