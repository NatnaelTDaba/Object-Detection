import sys
import os
import pickle
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as prebuilt_models
from torch.optim.lr_scheduler import _LRScheduler

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import ssl

from sklearn.metrics import classification_report

from config import *

ssl._create_default_https_context = ssl._create_unverified_context

def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        if args.pretrained:
            print("Loading pretrained model ...")
            net = pretrained_models.resnet18(pretrained=True)
            input_features = net.fc.in_features
            net.fc = nn.Linear(input_features, args.nclasses)
        else:
            print("Loading local model ...")
            from models.resnet import resnet18
            net = resnet18(args.nclasses)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(args.nclasses)
    elif args.net == 'resnet50':
        if args.pretrained:
            print("Loading pretrained model ...")
            net = pretrained_models.resnet50(pretrained=True)
            input_features = net.fc.in_features
            net.fc = nn.Linear(input_features, args.nclasses)
        else:
            print("Loading local model ...")
            from models.resnet import resnet50
            net = resnet50(args.nclasses)
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(args.nclasses)
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(args.nclasses)
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion Matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    fig = plt.figure(figsize=[14,14])
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    return fig

def save_object(filename, obj):
    
    if filename is None:
        print("Please provide filename.")
    
    f = open(DATA_DIRECTORY+filename, 'wb')
    pickle.dump(obj, f)
    f.close()

def load_object(filename):
    
    f = open(DATA_DIRECTORY+filename, 'rb')
    loaded = pickle.load(f)
    f.close()
        
    return loaded

import torch.nn as nn
import torch.nn.init as init


def init_weights(model):
    if isinstance(model, nn.Linear):
        if model.weight is not None:
            init.kaiming_uniform_(model.weight.data)
        if model.bias is not None:
            init.normal_(model.bias.data)
    elif isinstance(model, nn.BatchNorm1d):
        if model.weight is not None:
            init.normal_(model.weight.data, mean=1, std=0.02)
        if model.bias is not None:
            init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm2d):
        if model.weight is not None:
            init.normal_(model.weight.data, mean=1, std=0.02)
        if model.bias is not None:
            init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm3d):
        if model.weight is not None:
            init.normal_(model.weight.data, mean=1, std=0.02)
        if model.bias is not None:
            init.constant_(model.bias.data, 0)
    else:
        pass

def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]


def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def init_weights2(net):
    """the weights of conv layer and fully connected layers 
    are both initilized with Xavier algorithm, In particular,
    we set the parameters to random values uniformly drawn from [-a, a]
    where a = sqrt(6 * (din + dout)), for batch normalization 
    layers, y=1, b=0, all bias initialized to 0.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return net

def save_hparams(args, writer):
    
    with open(os.path.join(writer.get_logdir(), 'hparams.csv'), 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Parameter', 'Value'])
        for key, value in vars(args).items():
            csv_writer.writerow([key, value])

def get_stats(args):

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

    elif args.nclasses == 22:
        
        train_path = TRAIN_DIRECTORY_22CLASS
        val_path = VALIDATION_DIRECTORY_22CLASS
        mean = TRAIN_MEAN_22CLASS
        std = TRAIN_STD_22CLASS
    else:
        print("Invalid number of classes")
        sys.exit()

    return (train_path, val_path, mean, std)

def get_loss_weights(args):

    if args.nclasses == 24:
        return CLASS_WEIGHTS24
    elif args.nclasses == 22:
        return CLASS_WEIGHTS22

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        
        Formula can be found on section 3.1 of "Bag of Tricks for Image Classification with Convolutional Neural Networks" by Tong He, et al
        """
        #print("last_epoch var in WarmUpLR", self.last_epoch)
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias, 
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture
    
    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)
        
        else: 
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)
        
    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]

def save_report(targets, predictions, class_names, report_dir, epoch):

    with open(os.path.join(report_dir, 'epoch_' +str(epoch)+'_report.txt'), 'w') as f:

        print(classification_report(targets, predictions, target_names=class_names), file=f)
