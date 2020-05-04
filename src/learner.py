"""
Module for training model
"""

import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
from sklearn import metrics
import os
import copy
from tqdm import tqdm, trange
import logging as log
import time
from statistics import mean

import helper

class Learner():
    def __init__(self,
                 access_mode      = None,   # os access mode for created files
                 experiment_name  = None,   # name of experiment
                 model            = None,   # model
                 device           = None,   # device to run experiment
                 myio             = None,   # myio.IO object for loading data
                 save_dir         = None,   # directory to save results
                 max_steps        = 1e5,    # maximum number of update steps
                 best_int         = 500,    # interval for checking weights
                 verbose_int      = 100,    # interval for logging information
                 max_grad_norm    = 0.0,    # maximum gradients to avoid exploding gradients
                 optimizer        = None,   # optimizer for training
                 weight_decay     = 0.0,    # weight decay if using
                 lr               = 0.01,   # learning rate
                 eps              = 1e-8,   # epsilon to use for adam
                 accumulate_int   = 1,      # number of steps to accumulate gradients before stepping
                 batch_size       = 8,      # batch size
                 warmup_pct       = 0.0,    # percent of updates used to warm-up learning rate
                 save             = False,  # whether to save weights
                 patience         = 5,      # number of checks with no improvement before early stop
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
        self.best_int = best_int
        self.verbose_int = verbose_int
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.lr = lr
        self.eps = eps
        self.accumulate_int = accumulate_int
        self.batch_size = batch_size
        self.warmup_pct = warmup_pct
        self.save = save
        self.patience = patience
        
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
                   batch = None):
        """
        Labeled Data Batch: image, target, road_image, extra
        Unlabeled Data Batch: image, camera_index
        --------------------
        Returns:
        Dictionary of batch information with keys as model keywords

        """
        
        if self.labeled:
            inputs = {"images"       : batch[0],
                      "box_targets"  : batch[1],
                      "road_targets" : batch[2]}
        else:
            inputs = {"images"       : batch[0]}
        
        return inputs
        
    
    def train_step(self,
                   batch        = None,    # batch data
                   idx          = None,    # index of step
                   scheduler    = None,    # learning rate scheduler
                   optimizer    = None,    # optimizer for training
                   accumulated  = None,    # number of accumulated gradients so far
                   ):
        """
        Training for a single batch.
        
        --------------------
        Returns:
        l           - loss of the batch
        accumulated - accumulated 
        """
        self.model.train()
        inputs = self.pack_input(batch)
        
        # zero gradients
        if accumulated == 0:
            optimizer.zero_grad()
            
        # send data through model forward
        out = self.model(**inputs)
        # out[0] First element as loss
        # out[1] Second element as predicted bounding boxes( for evaluation.o.w.empty list)
        # out[2] Third element as road map
        # out[3] road_loss
        # out[4] box_loss (for train.o.w. 0)

        # model outputs loss as first entry of tuple
        l = out[0]
        road_l = out[3]
        box_l = out[4]
        
        # for multi-gpu
        if isinstance(self.model, nn.DataParallel):
            l = l.mean() # average on multi-gpu parallel training
            road_l = road_l.mean()
            box_l = box_l.mean()
        
        # calculate gradients through back prop
        l.backward()
        
        accumulated += 1
        
        # clip gradients
        if accumulated == self.accumulate_int:
            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            #take a step in gradient descent
            optimizer.step()
            scheduler.step()
            
            accumulated = 0
        
            # zero gradients
            optimizer.zero_grad()

        return l.detach().item(), accumulated, road_l.detach().item(), box_l.detach().item()
        
    def evaluate(self,
                 debug=False):
        """
        Evaluation model on task.
        
        ---------------------
        Returns:
        Average validation loss
        """            
        # puts model in evaluation mode
        self.model.eval()

        pred_boxes = []
        pred_roads = []

        road_ts = []
        box_ats = []

        # stop gradient tracking
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_dataloader, desc="Validation", mininterval=30)):
                inputs = self.pack_input(batch)
                # {"images": batch[0],
                #  "box_targets": batch[1],
                #  "road_targets": batch[2]}


                out = self.model(**inputs)
                # out[0] First element as loss
                # out[1] Second element as predicted bounding boxes( for evaluation.o.w.empty list)
                # out[2] Third element as road map
                # out[3] road_loss
                # out[4] box_loss(for train.o.w. 0)

                for road_map1, road_map2 in zip(pred_roads, inputs['road_targets']):
                    road_ts.append(helper.compute_ts_road_map(road_map1, road_map2))

                for boxes1, boxes2 in zip(pred_boxes, inputs['box_targets']):
                    box_ats.append(helper.compute_ats_bounding_boxes(boxes1, boxes2['bounding_box']))

                if i==1 and debug:
                    log.info('Debug')
                    break

        return mean(road_ts), mean(box_ats)
        
    def train(self,
              optimizer = None,  # optimizer to use for training
              scheduler = None,  # scheduler to use for training
              labeled = True,    # whether training on labeled data
              debug = True,
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
                                                    max_lr          = self.lr,
                                                    total_steps      = self.max_steps,
                                                    anneal_strategy = "linear",
                                                    cycle_momentum  = False,
                                                    pct_start       = self.warmup_pct,
                                                    )
        
        self.labeled = labeled

        cum_loss =  0.0
        cum_road_loss = 0.0
        cum_box_loss = 0.0
        best_val_road = 0.0
        best_val_image = 0.0
        best_iter = 0
        exp_log_dir = os.path.join(self.log_dir, self.experiment_name)
        stop = False
        no_improve = 0

        if self.save:
            # make directory for model weights for given task if doesn't exist
            if not os.path.exists(exp_log_dir):
                os.mkdir(exp_log_dir)
        
        best_path = os.path.join(exp_log_dir, 'best.pt')
        
        self.train_dataloader, self.val_dataloader = self.IO.load_dataloader(labeled=self.labeled)
        
        # set number of epochs based on number of iterations
        max_epochs = (self.max_steps // (len(self.train_dataloader.dataset)//(self.batch_size*self.accumulate_int))) + 1
        
        log.info("Training with {} iterations; ~ {} epochs".format(self.max_steps, max_epochs))

        # train
        global_step = 0
        accumulated = 0
        checked = False
        logged = False
        
        train_iterator = trange(0, int(max_epochs), desc = 'Epoch', mininterval=30)
        start = time.time()
        
        self.model.zero_grad()
        for epoch in train_iterator:
            epoch_iterator = tqdm(self.train_dataloader, desc='Train Iteration', mininterval=30)
            for step, batch in enumerate(epoch_iterator):
                iter_loss, accumulated, road_l, box_l = self.train_step(batch       = batch,
                                                                        idx         = global_step,
                                                                        scheduler   = scheduler,
                                                                        optimizer   = optimizer,
                                                                        accumulated = accumulated)
                cum_loss += iter_loss
                cum_road_loss += road_l
                cum_box_loss += box_l

                if accumulated == 0:
                    global_step += 1
                    checked = False
                    logged = False

                # check for best every best_int
                if global_step % self.best_int == 0 and not checked:
                    checked = True
                    log.info("="*40+" Evaluating on step: {}".format(global_step))
                    val_road, val_image = self.evaluate(debug)
                    
                    log.info("="*40+" Current Val Road {}, Box {}, Step = {} | Previous Best Loss {}, Box {}, Step = {}".format(
                        val_road,
                        val_image,
                        global_step,
                        best_val_road,
                        best_val_image,
                        best_iter))
                    
                    # check for best road ts
                    if val_road > best_val_road:
                        best_val_road = val_road

                    # check for best image ats
                    if val_image > best_val_image:
                        best_val_image = val_image

                    # save if either best evaluation metric saved
                    if val_road > best_val_road or val_image > best_val_image:
                        if self.save:
                            # for multi-gpu
                            if isinstance(self.model, nn.DataParallel):
                                best_state_dict = self.model.module.state_dict()
                            else:
                                best_state_dict = self.model.state_dict()
                            
                            torch.save(best_state_dict, best_path)
                            os.chmod(best_path, self.access_mode)
                    else:
                        no_improve += 1
                        if no_improve >= self.patience:
                            log.info('='*40+' Early stopping at step {} '.format(global_step)+'='*40)
                            stop = True


                # write to log every verbose_int
                if global_step % self.verbose_int == 0 and not logged:
                    logged = True
                    log.info('='*40+' Iteration {} of {} | Average Training Loss {:.6f} |'\
                             ' Best Val Road ts {} | Best Val Box ats {} | Best Iteration {} |'.format(
                                 global_step,
                                 self.max_steps,
                                 cum_loss/global_step,
                                 best_val_road,
                                 best_val_image,
                                 best_iter)
                        )
                    log.info('='*40+' Box Loss {} | Road Loss {} '.format(cum_road_loss/global_step, cum_box_loss/global_step)+'='*40)
                
                # break training if max steps reached (+1 to get max_step)
                if global_step > self.max_steps or stop:
                    epoch_iterator.close()
                    break
            if global_step > self.max_steps or stop:
                train_iterator.close()
                break
            
        # log finished results
        log.info('Finished | Average Training Loss {:.6f} |'\
                 ' Best Val Road ts {} | Best Val Box ats {} | Best Iteration {} | Time Completed {:.2f}s'.format(
                     cum_loss/global_step,
                     best_val_road,
                     best_val_image,
                     best_iter,
                     time.time()-start
                )
            )
        
        return {"avg_train_loss" : cum_loss/global_step,
                "avg_road_loss" : cum_road_loss/global_step,
                "avg_box_loss" : cum_box_loss/global_step,
                "best_val_road" : best_val_road,
                "best_val_image" : best_val_image,
                "best_val_step" : best_iter,
                "best_weights" : best_path,
                "current_step" : global_step,
                "max_step" : self.max_steps}