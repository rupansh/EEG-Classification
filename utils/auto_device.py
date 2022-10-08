import torch

def auto_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    return device