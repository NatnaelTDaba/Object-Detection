""" Defines gloabal variables for this project.

Author: Natnael Daba

"""
import os 
from datetime import datetime

WORKING_DIRECTORY = '/home/abhijit/nat/Object-Detection'
DATA_DIRECTORY = WORKING_DIRECTORY+'/data/'
UTILITIES_DIRECTORY = WORKING_DIRECTORY+'/data_utilities/'
TRAIN_DIRECTORY = DATA_DIRECTORY+'train_images_24classes_split/train'
VALIDATION_DIRECTORY = DATA_DIRECTORY+'train_images_24classes_split/val'
PLOTS_DIRECTORY = WORKING_DIRECTORY+'/runs/plots/'
CHECKPOINT_PATH = WORKING_DIRECTORY+'/checkpoint/'
WEAK_TRAIN_DIRECTORY = DATA_DIRECTORY+'train_images_top5_weak/train'
WEAK_VALIDATION_DIRECTORY = DATA_DIRECTORY+'train_images_top5_weak/val'

TRAIN_MEAN = [0.2559, 0.2135, 0.1866]
TRAIN_STD = [0.1558, 0.1394, 0.1320]

WEAK_TRAIN_MEAN = [0.2662, 0.2213, 0.1913]
WEAK_TRAIN_STD = [0.1572, 0.1395, 0.1303]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 32

NEW_SIZE = 32
NEW_SIZE_BIG = 224

CLASS_SIZE = 24

LOG_DIR = 'runs'

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

EPOCH = 500