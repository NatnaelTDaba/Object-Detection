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

BATCH_SIZE = 32

NEW_SIZE = 32

CLASS_SIZE = 24

LOG_DIR = 'runs'

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

EPOCH = 100