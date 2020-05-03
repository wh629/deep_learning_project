"""
Module with class io containing methods for importing and exporting data.
"""

import os
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import gzip
import json
import pandas as pd
import logging as log
import time
import numpy as np

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn
    
class IO:
    """
        
    """
    def __init__(self,
                 data_dir = None,   # name of the directory storing all tasks
                 batch_size=32,     # batch size for training
                 shuffle=True,      # whether to shuffle train sampling
                 split = 0.2,       # percent split for validation
                 ):
        
        self.data_dir =  data_dir
        assert os.path.exists(self.data_dir), "No data"

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split

        self.annotation_csv = os.path.join(data_dir, 'annotation.csv')
        self.unlabeled_scene_index = np.arange(106)
        self.labeled_scene_index = np.arange(106, 134)
        self.transform = torchvision.transforms.ToTensor()

        self.cached = os.path.join(data_dir, "cached")
        if not os.path.exists(self.cached):
            os.mkdir(self.cached)
        
# =============================================================================
# Methods to read data
# =============================================================================
    def load_dataloader(self,
                        labeled = True   # whether to get labeled data
                        ):
        """
        Load data and create dataloader
        
        Labeled Data Batch: image, target, road_image, extra
        Unlabeled Data Batch: image, camera_index

        --------------------
        Return:
        train_dl - dataloader for training data
        val_dl   - dataloader for validation data
        """
        train, val = self.read_data(labeled = labeled)

        train_dl = DataLoader(train,
                              batch_size = self.batch_size,
                              shuffle = self.shuffle,
                              collate_fn = collate_fn)

        val_dl = DataLoader(val,
                            batch_size = self.batch_size,
                            shuffle = False,
                            collate_fn = collate_fn)

        return train_dl, val_dl

    def read_data(self, 
                  labeled = True   # whether to read labeled data
                  ):
        """
        Load data and split to train/validation
        --------------------
        Return:
        train - training dataset
        val   - validation dataset
        """

        train = None
        val = None

        if labeled:
            subset = 'labeled'
            indices = self.labeled_scene_index
        else:
            subset = 'unlabeled'
            indices = self.unlabeled_scene_index

        cache_file = os.path.join(self.cached, subset+'.pt')

        if os.path.exists(self.cached_name):
            log.info('Loading from cached file: {}'.format(cache_file))
            loaded = torch.load(cache_file)
            train, val = (loaded['train'],
                          loaded['val'])
        else:
            if labeled:
                dataset = LabeledDataset(image_folder = self.data_dir,
                                         annotation_file = self.annotation_csv,
                                         scene_index = indices,
                                         transform = self.transform,
                                         extra_info = False)
            else:
                dataset = UnlabeledDataset(image_folder = self.data_dir,
                                           scene_index = indices,
                                           first_dim = 'sample',
                                           transform = self.transform)

            val_size = int(len(dataset)*self.split)
            train_size = int(len(dataset)-val_size)

            return torch.utils.data.random_split(dataset,[train_size, val_size])