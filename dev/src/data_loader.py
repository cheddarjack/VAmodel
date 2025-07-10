import os
import glob
import yaml
import numpy as np
import pandas as pd

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_and_pad_files(directory, features, label_col, fixed_length=2048):
    """
    Loads all parquet files in 'directory', sorts by datetime if present,
    and pads/truncates to a fixed length. Supports multi-column labels.
    """
    file_paths = glob.glob(os.path.join(directory, "**", "*.parquet"), recursive=True)
    X_list, y_list = [], []

    for f in file_paths:
        try:
            df = pd.read_parquet(f)
            if 'datetime' in df.columns:
                df = df.sort_values('datetime').reset_index(drop=True)

            data_array = df[features].values
            labels = df[label_col].values  # Handles single or multiple label columns

            # Pad or truncate data to fixed_length
            if len(df) < fixed_length:
                pad_len = fixed_length - len(df)
                # Pad features
                padded_data = np.pad(
                    data_array,
                    pad_width=((0, pad_len), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                # Pad labels
                if labels.ndim == 1:
                    padded_labels = np.pad(
                        labels,
                        pad_width=(0, pad_len),
                        mode="constant",
                        constant_values=0,
                    )
                else:
                    padded_labels = np.pad(
                        labels,
                        pad_width=((0, pad_len), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )
            else:
                padded_data = data_array[-fixed_length:]
                padded_labels = labels[-fixed_length:]

            X_list.append(padded_data)
            y_list.append(padded_labels)
        except Exception as e:
            print(f"[WARN] Could not process file {f}: {e}")

    # Convert to arrays
    try:
        X_array = np.array(X_list, dtype=np.float32)
        y_array = np.array(y_list, dtype=np.float32)
    except ValueError as e:
        print("[ERROR] Could not convert to numpy arrays. Possibly mismatched shapes.")
        print("Exception:", e)
        return np.array([]), np.array([])

    print(f"[INFO] Loaded {len(file_paths)} files. Final X shape={X_array.shape}, y shape={y_array.shape}")
    return X_array, y_array