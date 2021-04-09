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
TRAIN_DIRECTORY_22CLASS = DATA_DIRECTORY+'train_images_22classes_split/train'
VALIDATION_DIRECTORY_22CLASS = DATA_DIRECTORY+'train_images_22classes_split/val'
PLOTS_DIRECTORY = WORKING_DIRECTORY+'/runs/plots/'
CHECKPOINT_PATH = WORKING_DIRECTORY+'/checkpoint/'
WEAK_TRAIN_DIRECTORY = DATA_DIRECTORY+'train_images_top5_weak/train'
WEAK_VALIDATION_DIRECTORY = DATA_DIRECTORY+'train_images_top5_weak/val'
REPORTS_DIRECTORY = WORKING_DIRECTORY+'/reports/'

# mean and std of training dataset with 24 classes
TRAIN_MEAN = [0.2559, 0.2135, 0.1866]
TRAIN_STD = [0.1558, 0.1394, 0.1320]

# mean and std of training dataset with 5 classes
WEAK_TRAIN_MEAN = [0.2662, 0.2213, 0.1913]
WEAK_TRAIN_STD = [0.1572, 0.1395, 0.1303]

TRAIN_MEAN_22CLASS = [0.2532, 0.2125, 0.1867] 
TRAIN_STD_22CLASS = [0.1537, 0.1390, 0.1325]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LOG_DIR = 'runs'

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

CLASS_WEIGHTS24 = [ 71.80653021,   1.        ,  30.54456882, 192.10821382,
        58.51747419,  17.41484458,  35.97338867,  58.6339037 ,
       245.98831386,  51.90102149, 237.65645161, 131.91316025,
       115.65698587, 144.74165029, 288.91568627, 155.92275132,
       251.87521368, 299.48577236, 183.95380774, 205.21866295,
        66.10453118, 130.85879218, 116.3878357 , 133.83015441]
        
CLASS_WEIGHTS22 = [227.74390244, 146.75834971, 234.53689168, 352.35849057,
       158.0952381 , 303.04259635, 255.38461538, 292.94117647,
       303.65853659, 186.51685393, 208.07799443,   1.        ,
        67.02557201, 132.68206039, 118.00947867,  12.7181408 ,
        20.66390041, 249.41569282,  52.62416344,  30.97014925,
       133.75111907, 117.26844584]