"""
Module for training model
"""

import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
from sklearn import metrics
import transformers
import transformers.data.processors.squad as sq
import transformers.data.metrics.squad_metrics as sq_metrics
import os
import copy
from tqdm import tqdm, trange
import logging as log
import time

import helper

class Learner():
    def __init__(self,
                 access_mode,
                 experiment_name,
                 model,
                 device,
                 myio,
                 save_dir,
                 max_steps = 1e5,
                 log_int = 1e4,
                 best_int = 500,
                 verbose_int = 1000,
                 max_grad_norm = 1.0,
                 optimizer = None,
                 weight_decay = 0.0,
                 lr = 5e-3,
                 eps = 1e-8,
                 accumulate_int = 1,
                 batch_size = 8,
                 warmup_pct = 0.0
                 ):
        """
        Object to store learning. Used for fine-tuning.
        
        Data stored in myio.IO object called myio.
        """
        self.access_mode = access_mode
        self.experiment_name = experiment_name
        self.model = model.to(device)
        self.device = device
        self.IO = myio
        self.save_dir = save_dir
        self.max_steps = max_steps
        self.log_int = log_int
        self.best_int = best_int
        self.verbose_int = verbose_int
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.lr = lr
        self.eps = eps
        self.accumulate_int = accumulate_int
        self.batch_size = batch_size
        self.warmup_pct = warmup_pct
        
        # make directory for recorded weights if doesn't already exist
        self.log_dir = os.path.join(self.save_dir, 'logged')
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        
        # data
        self.train_dataloader = None
        self.val_dataloader = None
        
        # if multiple GPUs on single device
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(model)
            self.model.to(self.device)
    
    def get_optimizer(self):
        """
        Set optimizer for learner object using model.
        """
        # don't apply weight decay to bias and LayerNorm weights
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params"       : [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay" : self.weight_decay
            },
            {
                "params"       : [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay" : 0.0
            }
        ]
            
        optimizer = opt.AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.eps)
        
        return optimizer
    
    def pack_input(self,
                   ):
        """
        TO DO:
        
        pack input into dictionary
        """
        
        inputs = {}
        
        return inputs
        
    
    def train_step(self,
                   batch=None,
                   idx=None,
                   scheduler=None,
                   optimizer=None,
                   accumulated=None
                   ):
        """
        Training for a single batch.
        
        --------------------
        Returns:
        
        """
# =============================================================================
#         # TO DO: 
#         # unpack data and labels from batch and send to device
# =============================================================================
        inputs = self.pack_input() # TO IMPLEMENT
        
        # zero gradients
        if accumulated == 0:
            optimizer.zero_grad()
            
        # send data through model forward
        out = self.model(**inputs)
        
        # model outputs loss as first entry of tuple
        l = out[0]
        
        # for multi-gpu
        if isinstance(self.model, nn.DataParallel):
            l = l.mean() # average on multi-gpu parallel training
        
        # calculate gradients through back prop
        l.backward()
        
        accumulated += 1
        
        # clip gradients
        if accumulated == self.accumulate_int:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            #take a step in gradient descent
            optimizer.step()
            scheduler.step()
            
            accumulated = 0
        
            # zero gradients
            optimizer.zero_grad()
        
        return l.detach(), accumulated
        
    def evaluate(self,
                 model = None
                 ):
        """
        Evaluation model on task.
        
        ---------------------
        Returns:
        Average validation loss
        """
        # assign model if None
        if model is None:
            model = self.model
        else:
            if torch.cuda.is_available() and torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
                # multiple GPUs
                model = torch.nn.DataParallel(model)
            model.to(self.device)
            
        # puts model in evaluation mode
        model.eval()
        
        cum_loss = 0
        
        true_boxes = []
        true_roads = []

        pred_boxes = []
        pred_roads = []

        # stop gradient tracking
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
# =============================================================================
#                 # TO DO:
#                 # unpack data and labels from batch and send to device
# =============================================================================
                inputs = self.pack_input() # TO IMPLEMENT
                
                out = model(**inputs)
                
                l = out[0]
                
                if isinstance(self.model, nn.DataParallel):
                    l = l.mean()
                
                cum_loss += l.detach()
        
        #TODO: calculate ts road map and ats bounding boxes
        # might need to loop for road_ts
        for road_map1, road_map2 in zip(pred_roads, true_roads):
            road_ts = helper.compute_ts_road_map(road_map1, road_map2)

        for boxes1, boxes2 in zip(pred_boxes, pred_roads):
            box_ats = helper.compute_ats_bounding_boxes(boxes1, boxes2)

        return cum_loss/(i+1), road_ts, box_ats
        
    def train(self,
              optimizer = None,
              scheduler = None,
              ):
        """
        Fine-tune model on task
            
        --------------------
        Return: 
        logged_rln_paths - list of best RLN weight paths
        logged_f1        - list of best validation f1 scores
        best_path        - path of best weights
        """
        # set up learning rate scheduler
        if optimizer is None:
            optimizer = self.get_optimizer()
        
        if scheduler is None:
            scheduler = opt.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr = self.lr,
                                                    tota_steps=self.max_steps,
                                                    anneal_strategy="linear",
                                                    cycle_momentum=False,
                                                    pct_start=self.warmup_pct)
        
        cum_loss =  0.0
        best_val_loss = float("inf")
        best_val_road = 0.0
        best_val_image = 0.0
        best_iter = 0
        exp_log_dir = os.path.join(self.log_dir, self.experiment_name)
        
        # make directory for model weights for given task if doesn't exist
        if not os.path.exists(exp_log_dir):
            os.mkdir(exp_log_dir)
        
        best_path = os.path.join(exp_log_dir, 'best.pt')
        
# =============================================================================
#         # TO DO: load data
# =============================================================================
        self.train_dataloader = self.IO.load_dataloader(None)
        self.val_dataloader = self.IO.load_dataloader(None)
        
        # set number of epochs based on number of iterations
        max_epochs = (self.max_steps // (len(self.train_dataloader.dataset)//(self.batch_size*self.accumulate_int))) + 1
        
        log.info("Training with {} iterations ~ {} epochs".format(self.max_steps, max_epochs))

        # train
        global_step = 0
        accumulated = 0
        
        train_iterator = trange(0, int(max_epochs), desc = 'Epoch', mininterval=30)
        start = time.time()
        
        optimizer.zero_grad()
        for epoch in train_iterator:
            epoch_iterator = tqdm(self.train_dataloader, desc='Epoch Iteration', mininterval=30)
            for step, batch in enumerate(epoch_iterator):
                
                self.model.train()
                iter_loss, accumulated = self.train_step(batch=batch,
                                                         idx=global_step,
                                                         scheduler=scheduler,
                                                         optimizer=optimizer,
                                                         accumulated=accumulated)

                if accumulated == 0:
                    global_step += 1

                cum_loss += iter_loss
                
                # check for best every best_int
                if global_step % self.best_int == 0:
                    log.info("="*40+" Evaluating on step: {}".format(global_step))
                    val_results, val_road, val_image = self.evaluate()
                    
                    log.info("="*40+" Current Val Loss {}, Step = {} | Previous Best Loss {}, Step = {}".format(
                        val_results,
                        global_step,
                        best_val_loss,
                        best_iter))
                    
                    if val_results < best_val_loss:
                        best_val_loss = val_results
                        best_iter = global_step
                        
                        # for multi-gpu
                        if isinstance(self.model, nn.DataParallel):
                            best_state_dict = self.model.module.state_dict()
                        else:
                            best_state_dict = self.model.state_dict()
                        
                        torch.save(best_state_dict, best_path)
                        os.chmod(best_path, self.access_mode)
                    
                    if val_road > best_val_road:
                        best_val_road = val_road

                    if val_image > best_val_image:
                        best_val_image = val_image

                # write to log every verbose_int
                if global_step % self.verbose_int == 0:
                    log.info('='*40+' Iteration {} of {} | Average Training Loss {:.6f} |'\
                             ' Best Val F1 {} | Best Iteration {} |'.format(
                                 global_step,
                                 self.max_steps,
                                 cum_loss/global_step,
                                 best_val_loss,
                                 best_iter
                        )
                    )
                
                global_step += 1
                
                # break training if max steps reached (+1 to get max_step)
                if global_step > self.max_steps+1:
                    epoch_iterator.close()
                    break
            if global_step > self.max_steps+1:
                train_iterator.close()
                break
            
        # log finished results
        log.info('Finished | Average Training Loss {:.6f} |'\
                 ' Best Val F1 {} | Best Iteration {} | Time Completed {:.2f}s'.format(
                     cum_loss/global_step,
                     best_val_loss,
                     best_iter,
                     time.time()-start
                )
            )
        
        return {"best_val_loss" : best_val_loss, "best_val_road" : best_val_road, "best_val_image" : best_val_image, "best_val_step" : best_iter, "best_weights" : best_path}