# -------------------------------------------------------------------------
# ------ Defines a Dataset class compatible with PyTorch DataLoader. ------
# -------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # X should be shape (num_samples, seq_length, num_features)
        # y should be shape (num_samples, seq_length)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return one sample: X: (seq_length, num_features) and y: (seq_length,)
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


# Why we do this: PyTorch’s training loop expects a Dataset that can be automatically batched, shuffled, etc. 
#                 The DataLoader uses TimeSeriesDataset to feed the model data in batches.