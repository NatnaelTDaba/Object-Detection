import torch 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchsampler import ImbalancedDatasetSampler


import PIL

from config import *

def get_loader(args):
    """ Returns a PyTorch data loader object
    

    """
    loader = {}
    datasets = {}

    if args.nclasses == 5:
        train_path = WEAK_TRAIN_DIRECTORY
        val_path = WEAK_VALIDATION_DIRECTORY
        mean = WEAK_TRAIN_MEAN
        std = WEAK_TRAIN_STD
    elif args.nclasses == 24:
        train_path = TRAIN_DIRECTORY
        val_path = VALIDATION_DIRECTORY
        mean = TRAIN_MEAN
        std = TRAIN_STD
    else:
        print("Invalid number of classes")
        sys.exit()

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
                                        transforms.Normalize(mean, std)])
                        }

    train_set = ImageFolder(root=train_path, transform=transformations['training_transform'], )
    validation_set = ImageFolder(root=val_path, transform=transformations['validation_transform'])

    if args.balanced:
        print("Balanced training")
        loader['training'] = DataLoader(train_set, sampler=ImbalancedDatasetSampler(train_set), batch_size=args.b)
        loader['validation'] = DataLoader(validation_set, sampler=ImbalancedDatasetSampler(validation_set), batch_size=args.b)
    else:
        print("Imbalanced training")
        loader['training'] = DataLoader(train_set, batch_size=args.b, shuffle=True)
        loader['validation'] = DataLoader(validation_set, batch_size=args.b)

    datasets['training'] = train_set
    datasets['validation'] = validation_set

    return datasets, loader
