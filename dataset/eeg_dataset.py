import gc
import os
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

from utils.raw_loaders import load_training_data

INPUT_SZ = 2**10
CHUNK_SZ = 1000

def resample_data(gt):
    mean_v = []
    thresh = 0.01
    idx = []

    for i in range(len(gt)):
        for j in range(0, gt[i].shape[1], CHUNK_SZ):
            mean_v.append(np.mean(gt[i][:, j:min(gt[i].shape[1],j+CHUNK_SZ)]))
            if mean_v[-1] > thresh:
                idx.extend([(i, k) for k in range(j, min(gt[i].shape[1],j+CHUNK_SZ))])
    
    del mean_v
    gc.collect()

    return idx

def load_or_calc_ms(data): 
    if os.path.isfile("./data/mean.npy"):
        mean = np.load("./data/mean.npy")
    else:
        mean = np.mean(np.concatenate(data, axis=1), axis=1, keepdims=1)
        np.save("./data/mean.npy", mean)
        gc.collect()
    
    if os.path.isfile("./data/std.npy"):
        std = np.load("./data/std.npy")
    else:
        std = np.std(np.concatenate(data, axis=1), axis=1, keepdims=1)
        np.save("./data/std.npy", std)
        gc.collect()
    
    return mean, std

class EEGDataset(Dataset):
    def __init__(self, data, gt, soft_label=True, training=True):
        self.data: np.array = data
        self.gt = gt
        self.train = training
        self.soft_label = soft_label
        self.eps = 1e-7
        if self.train:
            self.idx = resample_data(self.gt)
        else:
            self.idx = [(i, j) for i in range(len(data)) for j in range(data[i].shape[1])]
        
        self.mean, self.std = load_or_calc_ms(data)

        for d in data:
            d = (d - self.mean)/(self.std+self.eps)

    def __getitem__(self, i):
        i, j = self.idx[i]
        raw, lab = self.data[i][:,max(0, j-INPUT_SZ+1):j+1], self.gt[i][:,j]

        padding = INPUT_SZ - raw.shape[1]
        if padding:
            raw = np.pad(raw, ((0, 0),(padding,0)), 'constant', constant_values=0)

        raw, lab = torch.from_numpy(raw.astype(np.float32)), torch.from_numpy(lab.astype(np.float32))

        if self.soft_label:
            lab[lab < 0.02] = 0.02
        
        return raw, lab

    def __len__(self):
        return len(self.idx)

def training_eeg_loader(batch_size: int):
    print("Loading Data...")
    ((ts, gt), (_, _)) = load_training_data()

    print("Init Dataset")
    ds = EEGDataset(ts, gt)
    return DataLoader(ds, batch_size=batch_size, num_workers=1, shuffle=True)