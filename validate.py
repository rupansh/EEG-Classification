import torch
from torch.utils.data import DataLoader
from utils.raw_loaders import load_training_data
from dataset.eeg_dataset import EEGDataset
from models.inception import default_inception
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics

device = "cuda:0" if torch.cuda.is_available() else "cpu"
#device = "cpu"

print("Device:", device)

model = default_inception() 
model.load_state_dict(torch.load("./inception.pt"))

model.to(device)

labels = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff',
       'Replace', 'BothReleased']

print("Loading Data...")
((_, _), (vts, vgt))  = load_training_data()
print("Init Dataset")
ds = EEGDataset(vts, vgt, False, False)
loader = DataLoader(ds, batch_size=256, num_workers=1)

model.eval()
y_pred = []
with torch.no_grad():
    for x, _ in tqdm(loader):
        x = x.to(device)
        pred = model(x).detach().cpu().numpy()
        y_pred.append(pred)

def plot_roc(y_true, y_pred):
    fig, axs = plt.subplots(3, 2, figsize=(15,13))
    for i, label in enumerate(labels):
        fpr, tpr, _ = metrics.roc_curve(y_true[i], y_pred[i])
        ax = axs[i//2, i%2]
        ax.plot(fpr, tpr)
        ax.set_title(label+" ROC")
        ax.plot([0, 1], [0, 1], 'k--')

    plt.show()
    
y_pred = np.concatenate(y_pred, axis=0)
vgt = np.concatenate(vgt, axis=1)
plot_roc(vgt, y_pred.T)
print('auc roc: ', metrics.roc_auc_score(vgt.T, y_pred))
