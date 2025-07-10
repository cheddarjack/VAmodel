
import os
import glob
import numpy as np
import pandas as pd
import torch


class BalancedWindowDataset(torch.utils.data.Dataset):
    def __init__(self, root, features, labels, seq_len):
        self.seq_len   = seq_len
        self.features  = features
        self.label_col = labels if isinstance(labels, list) else [labels]

        self.blocks = []          # (file_path, start_row)
        for fp in glob.glob(os.path.join(root, '**/*.parquet'), recursive=True):
            df = pd.read_parquet(fp)
            y  = df[self.label_col].values
            # find every index whose *window* (i … i+2047) contains ≥1 non‑zero
            nz = np.where(y.any(axis=1) if y.ndim == 2 else y != 0)[0]
            for idx in nz:
                start = max(0, idx - seq_len//2)           # centre window
                self.blocks.append((fp, start))
        print(f'[INFO] BalancedWindowDataset: {len(self.blocks)} windows')

    def __len__(self):  return len(self.blocks)

    def __getitem__(self, i):
        fp, start = self.blocks[i]
        df  = pd.read_parquet(fp)
        end = start + self.seq_len
        block_F = df[self.features].iloc[start:end].to_numpy(float)
        block_L = df[self.label_col].iloc[start:end].to_numpy(float)

        # pad/truncate to exact len = seq_len
        if len(block_F) < self.seq_len:
            pad = self.seq_len - len(block_F)
            block_F = np.pad(block_F, ((0,pad),(0,0)), 'constant')
            block_L = np.pad(block_L, ((0,pad),(0,0)), 'constant')
        return torch.from_numpy(block_F), torch.from_numpy(block_L)
