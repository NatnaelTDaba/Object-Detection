import torch 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split


def get_loader(**kwargs):
    """ Returns a PyTorch data loader object
    

    """
    loader = {}
    dataset = ImageFolder(root=kwargs['data_path'],transform=kwargs['transform'])
    trainset, valset = random_split(dataset, [189829, 81355], torch.Generator().manual_seed(42))
    loader['train'] = DataLoader(trainset, batch_size=kwargs['batch_size'], shuffle=kwargs['shuffle'])
    loader['val'] = DataLoader(valset, batch_size=kwargs['batch_size'])

    return loader
