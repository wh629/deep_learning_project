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
import transformers.data.processors.squad as sq
import time
import numpy as np

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn
    
class IO:
    """
        
    """
    def __init__(self,
                 data_dir,                # name of the directory storing all tasks
                 batch_size=32,           # batch size for training
                 shuffle=True,            # whether to shuffle train sampling
                 split = 0.2,
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
        
        Labeled Data Batch: sample, target, road_image, extra
        Unlabeled Data Batch: image, camera_index

        --------------------
        Return:
        train_dl - dataloader for training data
        val_dl   - dataloader for validation data
        """
        train, val = self.read_data(labeled = labeled)

        train_dl = DataLoader(train,
                              batch_size = self.batch_size,
                              shuffle = True,
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
                                         extra_info = True)
            else:
                dataset = UnlabeledDataset(image_folder = self.data_dir,
                                           scene_index = indices,
                                           first_dim = 'sample',
                                           transform = self.transform)

            val_size = int(len(dataset)*self.split)
            train_size = int(len(dataset)-val_size)

            return torch.utils.data.random_split(dataset,[train_size, val_size])


# =============================================================================
#         start = time.time()
#         
#         # make sure use is either train or dev
#         try:
#             self.uses.index(use)
#         except ValueError:
#             log.info("Use {} not supported.".format(use))
#         
#         # name cached file
#         cache_file = os.path.join(self.cache_dir,
#                                   "cached_{}_{}_{}.pt".format(
#                                       use,
#                                       task,
#                                       self.max_seq_length))
#         
#         if os.path.exists(cache_file):
#             # load dataset from cached file
#             
#             log.info('Loading {} {} from cached file: {}'.format(
#                 task, use, cache_file))
#             loaded = torch.load(cache_file)
#             features, dataset, examples = (
#                 loaded['features'],
#                 loaded['dataset'],
#                 loaded['examples']
#                 )
#         else:
#             # get dataset from .json file with correct formatting
#             data_dir = os.path.join(self.data_dir, task)
#             
#             if use == 'train':
#                 training = True
#                 examples = self.processor.get_train_examples(data_dir)
#             elif use == 'dev':
#                 training = False
#                 examples = self.processor.get_dev_examples(data_dir)
#                     
#             # convert data to squad objects
#             features, dataset = sq.squad_convert_examples_to_features(
#                 examples=examples,
#                 tokenizer=self.tokenizer,
#                 max_seq_length=self.max_seq_length,
#                 doc_stride=self.doc_stride,
#                 max_query_length=self.max_query_length,
#                 is_training= training,
#                 return_dataset = 'pt'
#                 )
#                     
#             # save cached
#             if self.cache:
#                 log.info('Saving {} processed data into cached file: {}'.format(len(dataset), cache_file))
#                 torch.save({'features': features, 'dataset': dataset, 'examples': examples}, cache_file)
#                 
#         # wrap dataset with DataLoader object
#         if use == 'train' and self.shuffle:
#             sampler = RandomSampler(dataset)
#         else:
#             sampler = SequentialSampler(dataset)
#             
#         dl = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)
#         
#         log.info("Task {} took {:.6f}s".format(task, time.time()-start))
# =============================================================================
        dl = None
        return dl   
    
