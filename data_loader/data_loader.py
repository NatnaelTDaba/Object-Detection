import torch 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from config import *

def get_loader(**kwargs):
    """ Returns a PyTorch data loader object
    

    """
    loader = {}

    train_set = ImageFolder(root=TRAIN_DIRECTORY, transform=kwargs['training_transform'], )
    validation_set = ImageFolder(root=VALIDATION_DIRECTORY, transform=kwargs['validation_transform'])

    loader['training'] = DataLoader(train_set, batch_size=kwargs['batch_size'], shuffle=True)
    loader['validation'] = DataLoader(validation_set, batch_size=kwargs['batch_size'])

    return loader
