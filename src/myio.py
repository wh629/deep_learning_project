"""
Module with class io containing methods for importing and exporting data.
"""

import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import gzip
import json
import pandas as pd
import logging as log
import transformers.data.processors.squad as sq
import time
    
class IO:
    """
        
    """
    def __init__(self,
                 data_dir,                # name of the directory storing all tasks
                 batch_size=32,           # batch size for training
                 shuffle=True,            # whether to shuffle train sampling
                 ):
        
        self.data_dir =  data_dir
        assert os.path.exists(self.data_dir), "No data"
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
# =============================================================================
# Methods to read data
# =============================================================================
    def load_dataloader(self,
                        use    # use case. either train or dev
                        ):
        """
        Load data and create dataloader
        
        --------------------
        Return:
        dl - dataloader
        """
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
    
