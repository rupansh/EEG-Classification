import torch
from models.resnet1d import default_resnet
from dataset.eeg_dataset import training_eeg_loader

from utils.trainer import start_training

model = default_resnet() 

loader = training_eeg_loader(256)

start_training(model, loader)

torch.save(model.state_dict(), "resnet.pt")
