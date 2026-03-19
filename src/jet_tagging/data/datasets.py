from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from jet_tagging.config import DATA_DIR


class JetImageDataset(Dataset):

    def __init__(self, h5_path, n_sample=10000):

        self.file = h5py.File(h5_path, 'r')
        self.n_sample = n_sample
        self.images = self.file['images'][:n_sample]
        self.labels = self.file['labels'][:n_sample].astype('int64')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img = self.images[idx]

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label


class PersistenceImagesDataset(Dataset):

    def __init__(self, h5_path, pi_key, n_sample=10000):

        self.file = h5py.File(h5_path, 'r')
        self.n_sample = n_sample
        self.images = self.file[pi_key][:n_sample]
        self.labels = self.file['labels'][:n_sample].astype('int64')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img = self.images[idx]

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label
    

class PersistenceImagesStackedDataset(Dataset):

    def __init__(self, h5_path, n_sample=10000):

        self.file = h5py.File(h5_path, 'r')

        self.H0 = self.file['pi_H0'][:n_sample]
        self.H1 = self.file['pi_H1'][:n_sample]

        self.labels = self.file['labels'][:n_sample].astype('int64')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        h0 = self.H0[idx]
        h1 = self.H1[idx]

        # (2,40,40)
        img = torch.tensor(np.array([h0, h1]), dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label
    