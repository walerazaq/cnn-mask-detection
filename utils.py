import torch
import os
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import glob as glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time

def train_valid_test_split(img_dir=None, split=0.15):
    
    files = os.listdir(img_dir)

    all_img = files
    
    random.shuffle(all_img)
    
    len_imgs = len(all_img)
    
    trainTest_split = int((1-split)*len_imgs)
    
    trainVal_df = all_img[:trainTest_split]
    test_df = all_img[trainTest_split:]
    
    lenTV_df = len(trainVal_df)
    
    trainVal_split = int((1-split)*lenTV_df)
    
    train_df = trainVal_df[:trainVal_split]
    valid_df = trainVal_df[trainVal_split:]
    
    return train_df, valid_df, test_df

def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.RandomRotate90(0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=5, p=0.2),
        A.Blur(blur_limit=5, p=0.2),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

class Averager:
    """""
    this class keeps track of the training and validation loss values...
    and helps to get the average for each epoch as well
    """""
    
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
        
def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))