from torch.utils.tensorboard.writer import SummaryWriter
from models.basic_net import NNet
from dataset.eeg_dataset import training_eeg_loader

writer = SummaryWriter("torchlogs/")
loader = training_eeg_loader(512)
model = NNet()
writer.add_graph(model, next(iter(loader))[0])
writer.close()

