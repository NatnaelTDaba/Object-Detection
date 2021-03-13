import torch 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import PIL

from config import *

def get_loader(args):
    """ Returns a PyTorch data loader object
    

    """
    loader = {}
    datasets = {}

    if args.pretrained:
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
        print("Using Imagenet statistics")
    else:
        mean = TRAIN_MEAN
        std = TRAIN_STD
    transformations = {'training_transform':transforms.Compose([
                                        transforms.Resize((args.resize, args.resize)),
                                        transforms.ColorJitter(hue=.05, saturation=.05, brightness=0.09),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)]),

                        'validation_transform':transforms.Compose([
                                        transforms.Resize((args.resize, args.resize)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])}

    train_set = ImageFolder(root=TRAIN_DIRECTORY, transform=transformations['training_transform'], )
    validation_set = ImageFolder(root=VALIDATION_DIRECTORY, transform=transformations['validation_transform'])

    loader['training'] = DataLoader(train_set, batch_size=args.b, shuffle=True)
    loader['validation'] = DataLoader(validation_set, batch_size=args.b)

    datasets['training'] = train_set
    datasets['validation'] = validation_set

    return datasets, loader
