import os
import time
import gc
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data_loader import load_config, load_and_pad_files
from src.dataset import TimeSeriesDataset
from src.model import Longformer, Transformer, TCN
from src.train_utils import train

def selective_minmax_scaling_y(Y, scale_indices):

    Y_scaled = np.copy(Y)
    n_sequences, seq_length, n_labels = Y.shape

    # Ensure scale_indices is a list.
    if isinstance(scale_indices, int):
        scale_indices = [scale_indices]
        
    for i in range(n_sequences):
        seq_y = Y[i]  # shape: (seq_length, n_labels)
        # Create a mask for non-padded rows.
        mask = ~np.all(seq_y == 0, axis=1)
        if np.sum(mask) > 0:
            # Scale each designated label column.
            for col in scale_indices:
                feat_vals = seq_y[mask, col]
                feat_min = feat_vals.min()
                feat_max = feat_vals.max()
                denom = feat_max - feat_min if (feat_max - feat_min) != 0 else 1
                Y_scaled[i, :, col] = (seq_y[:, col] - feat_min) / denom
                # print(f"Label {col} in file {i} scaled with min: {feat_min}, max: {feat_max}")
    return Y_scaled

def per_file_minmax_scaling(X, scale_indices=None, scale_groups=None):
    X_scaled = np.empty_like(X)
    n_sequences, seq_length, num_features = X.shape
    
    # If no independent indices provided, scale all features individually.
    if scale_indices is None:
        scale_indices = list(range(num_features))
    
    # If no groups provided, use empty list.
    if scale_groups is None:
        scale_groups = []

    for i in range(n_sequences):
        seq = X[i]  # shape: (seq_length, num_features)
        # Create a mask that ignores rows that are all zeros (assumed to be padded)
        mask = ~np.all(seq == 0, axis=1)
        if np.sum(mask) > 0:
            seq_scaled = np.copy(seq)
            
            # First, scale features that are to be scaled independently.
            # Skip features that are part of any scaling group.
            for j in scale_indices:
                # If this feature is in any group, skip it.
                if any(j in group for group in scale_groups):
                    continue
                feat_vals = seq[mask, j]
                feat_min = feat_vals.min()
                feat_max = feat_vals.max()
                denom = feat_max - feat_min if feat_max - feat_min != 0 else 1
                seq_scaled[:, j] = (seq[:, j] - feat_min) / denom

            # Now, scale each group of features together.
            for group in scale_groups:
                if len(scale_groups) == 1:
                    # Skip groups with only one feature.
                    continue
                # Extract all values for the features in this group
                seq_masked = seq[mask]
                if seq_masked.size(0) > 0 and group < seq_masked.size(1):
                    group_vals = seq_masked[:, group]
                else:
                    group_vals = torch.zeros((seq_masked.size(0),), device=seq.device)  # or your fallback
                group_min = group_vals.min()
                group_max = group_vals.max()
                denom = group_max - group_min if group_max - group_min != 0 else 1
                # Apply the same scaling to every feature in the group.
                seq_scaled[:, group] = (seq[:, group] - group_min) / denom

            # Reset padded rows to zero if needed.
            seq_scaled[~mask, :] = 0
            X_scaled[i] = seq_scaled
        else:
            X_scaled[i] = seq
    return X_scaled

def count_occurrences(data, values_to_count):
        """
        Flatten data, ignore zeros, then count exact occurrences for the given list of values.
        Returns a dict {value: (count, percentage)}.
        """
        # Flatten (n_sequences x seq_length x num_features or num_labels -> 1D)
        data_flat = data.flatten()
        # Ignore zeros
        data_nonzero = data_flat[data_flat != 0]
        total_nonzero = len(data_nonzero)

        occurrences = {}
        for val in values_to_count:
            count_val = np.sum(data_nonzero == val)
            pct_val = (count_val / total_nonzero * 100.0) if total_nonzero != 0 else 0.0
            occurrences[val] = (count_val, pct_val)
        return occurrences

def get_multipliers(data, increase,values_to_count):
        """
        Returns a dict mapping each value in `values_to_count` to the multiplier
        that would make its occurrences ~20% of all nonzero data points.
        """
        data_flat = data.flatten()
        data_nonzero = data_flat[data_flat != 0]
        total_nonzero = len(data_nonzero)
        target_count = int(increase * total_nonzero)

        multipliers = {}
        for val in values_to_count:
            count_val = np.sum(data_nonzero == val)
            if count_val == 0:
                # No occurrences; define behavior as you prefer (e.g. None).
                multipliers[val] = None
            else:
                multipliers[val] = target_count / count_val
        return multipliers


# ============================
#           Main Function
# ============================

def main():

    config_path = "/home/cheddarjackk/Developer/VAmodel/dev/config.yaml"
    config = load_config(config_path)

    # Sequence Prep

    X, y = load_and_pad_files(
        directory=config['data']['training_data_dir_2'],
        features=config['data']['features'],
        label_col=config['data']['label_column'],
        fixed_length=config['data']['seq_length']
    )
    # -------------------------------------------------------------
    #  Build an inverse‑frequency weight for every label value
    # -------------------------------------------------------------
    allowed_vals = config["data"]["allowed_values"]        # = [0, 2, 3, …]
    mapping       = {v: i for i, v in enumerate(allowed_vals)}

    targets_flat  = torch.as_tensor(y.reshape(-1), dtype=torch.long)   # ALL labels

    counts = torch.zeros(len(allowed_vals))
    for v, idx in mapping.items():                         # count each class
        counts[idx] = (targets_flat == v).sum()

    weights = 1.0 / (counts + 1e-9)                        # inverse‑freq
    weights = weights / weights.sum() * len(weights)       # normalise
    bg_scale = config["data"].get("bg_weight", 0.05)       # push class “0” down
    weights[mapping[0]] *= bg_scale

    #  multiplier_dict is what train_utils expects
    multiplier_dict = {v: float(weights[mapping[v]]) for v in allowed_vals}

    print(f"Loaded {len(X)} sequences.")
    # ============================
    #          Data Conversion
    # ============================
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Data Integrity

    if np.isnan(X).any() or np.isnan(y).any():
        print("NaNs found in the input data! Exiting.")
        print(f"X NaNs: {np.isnan(X).sum()}, y NaNs: {np.isnan(y).sum()}")
        return
    if np.isinf(X).any() or np.isinf(y).any():
        print("Infs found in the input data! Exiting.")
        return

    # Scaling

    print("Before Scaling:")
    print(f"X min: {X.min()}, X max: {X.max()}, mean: {X.mean():.4f}")
    print(f"y min: {y.min()}, y max: {y.max()}, mean: {y.mean():.4f}")

    scale_indices = [0,1]          # Scale these features individually.
    scale_groups = [[]]                 # Scale features 6, 7, 8 together.
    X  = per_file_minmax_scaling(X, scale_indices=scale_indices, scale_groups=scale_groups)

    if len(config['data']['label_column']) > 3:
        scale_label_indices = [1]  
        y = selective_minmax_scaling_y(y, scale_label_indices)

    print("After Scaling:")
    print(f"X min: {X.min()}, X max: {X.max()}, mean: {X.mean():.4f}")
    print(f"y min: {y.min()}, y max: {y.max()}, mean: {y.mean():.4f}")

    print(f"Data shapes - X: {X.shape}, y: {y.shape}")

    values_of_interest = [2, -2, 3, -3, 4, -4, 5, -5]
    increase = config['data']['increase']  # e.g., 0.2 for 20%
    occ = count_occurrences(y, values_of_interest)
    multiplier_dict = get_multipliers(y, increase, values_of_interest)

    if config['data'].get('multipliers', 'on') == 'off':
        multiplier_dict = {val: 1.0 for val in values_of_interest}
    
    print(f"Occurrences of values (ignoring zeros) and Multipliers needed to bring each value to ~{config['data']['increase']*100}%")
    for val in values_of_interest:
        cnt, pct = occ.get(val, (0, 0.0))
        multiplier = multiplier_dict.get(val, None)
        print(f"  Value {val:>2}: Count = {cnt}, Percentage = {pct:.2f}%, Multiplier = {multiplier}")
    
    # ============================
    #      Train/Validation Split
    # ============================
    split_idx = max(1, int(len(X) * config['data']['train_val_split']))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    mask = (y.sum(axis=(1, 2)) != 0)     # keep sequences that contain ≥1 non‑zero
    X, y = X[mask], y[mask]

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True          # ↓ host→GPU copy speed-up
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    num_batches = len(train_loader)
    n_sequences, seq_length, _ = X_train.shape  # Assuming you stored X_train before creating dataset
    total_rows = n_sequences * seq_length
    print(f"Total Batches per epoch: {num_batches}")
    print(f"Total rows processed per epoch: {total_rows}")

    # Model Initialization

    model_type = config['model']['type'].lower()  # e.g. "longformer" or "transformer"
    print(f"Model type: {model_type}")

    if model_type == "longformer":
        model = Longformer(
            d_model=config['model']['d_model'],
            num_hidden_layers=config['model']['num_encoder_layers'],
            num_attention_heads=config['model']['nhead'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            num_features=len(config['data']['features']),
            attention_window=config['model'].get('attention_window', 2048),
            global_positions=None,
            output_size=config['model']['output_size'],
            num_labels = len(config['data']['label_column'])
        )
    elif model_type == "transformer":
        model = Transformer(
            d_model=config['model']['d_model'],
            num_hidden_layers=config['model']['num_encoder_layers'],
            num_attention_heads=config['model']['nhead'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            num_features=len(config['data']['features']),
            output_size=config['model']['output_size'],
            max_seq_len=2048,
            num_labels = len(config['data']['label_column'])
        )
    elif model_type == "tcn":
        model = TCN(
            d_model=config['model']['d_model'],
            num_levels=config['model'].get('tcn_num_levels', 4),
            kernel_size=config['model'].get('tcn_kernel_size', 3),
            dropout=config['model']['dropout'],
            num_features=len(config['data']['features']),
            output_size=config['model']['output_size'],
            tcn_padding=config['model'].get('tcn_padding', 'same'),
            tcn_channel_growth=config['model'].get('tcn_channel_growth', False),
            num_labels = len(config['data']['label_column']),
            max_dilation=config['model'].get('max_dilation', 128),
            num_values=len(config['data']['allowed_values']),
        )
    else:
        raise ValueError(f"Unsupported model type in config: {model_type}")

    def initialize_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    model.apply(initialize_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.cuda.empty_cache()
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    print(f"Model device: {next(model.parameters()).device}")
    # ============================
    #            Training
    # ============================

    train(model, train_loader, val_loader, config, multiplier_dict)
    print("Training completed.")

if __name__ == "__main__":
    main()