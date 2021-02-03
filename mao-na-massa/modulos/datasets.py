from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

class Bicicletas(Dataset):

    def __init__(self, data_frame):
        self.dados = data_frame.to_numpy()

    def __getitem__(self, idx):
        features = self.dados[idx][:-1]
        labels = self.dados[idx][-1:]

        features = torch.from_numpy(features.astype(np.float32))
        labels = torch.from_numpy(labels.astype(np.float32))

        return features, labels

    def __len__(self):
        return len(self.dados)

