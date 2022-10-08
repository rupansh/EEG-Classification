import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from utils.auto_device import auto_device

def start_training(model, loader, epochs=1, device=auto_device()):
    model.to(device)
    loss_fn = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.99))

    loss_h, trainl = [], []
    print("Starting training..")
    model.train()
    for e in range(epochs):
        p_bar = tqdm(loader)
        for i, (x, y) in enumerate(p_bar):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            trainl.append(loss.item())
            p_bar.set_description(f"Loss: {trainl[-1]}")
            #print("Loss:", trainl[-1], "N:", i)

            if not i % 50:
                loss_h.append(np.mean(trainl))
                trainl.clear()
        print(f'epoch {e+1}/{epochs}, loss {loss_h[-1]}')
        loss_h.clear() 
