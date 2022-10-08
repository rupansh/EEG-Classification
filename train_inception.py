import torch
from models.inception import default_inception
from dataset.eeg_dataset import training_eeg_loader

from utils.trainer import start_training

model = default_inception()

loader = training_eeg_loader(1024)

start_training(model, loader)

torch.save(model.state_dict(), "inception.pt")
