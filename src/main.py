"""
Main run script to execute experiments and analysis
"""

import torch
import os
import logging as log
from datetime import datetime as dt
import random
import numpy as np
import sys
import time
import json

# =============== Self Defined ===============
import myio                        # module for handling import/export of data
import learner                     # module for fine-tuning
import model                       # module to define model architecture
from args import args, check_args  # module for parsing arguments for program

def main():
    """
    Main method for experiment
    """ 
    start = time.time()
    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser = args.parse_args()

    if parser.run_log == 'log':
        parser.run_log = os.path.join(parser.save_dir, 'log')

    if not os.path.exists(parser.run_log):
        os.mkdir(parser.run_log)

    # run some checks on arguments
    check_args(parser)
    
    # format logging
    log_name = os.path.join(parser.run_log, '{}_run_log_{}.log'.format(
        parser.experiment,
        dt.now().strftime("%Y%m%d_%H%M")
        )
    )
    log.basicConfig(filename=log_name,
                    format='%(asctime)s | %(name)s -- %(message)s',
                    level=log.DEBUG)
    os.chmod(log_name, parser.access_mode)
    
    # set devise to CPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info("Device is {}".format(device))
    
    # set seed for replication
    random.seed(parser.seed)
    np.random.seed(parser.seed)
    torch.manual_seed(parser.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(parser.seed)
    
    log.info("Starting experiment {} on {}".format(
        parser.experiment,
        device))
    
    data_handler = myio.IO(data_dir    = parser.data_dir,                # directory storing data
                           batch_size  = parser.batch_size,              # batch size
                           shuffle     = not parser.no_shuffle,          # whether to shuffle training data
                           split       = parser.val_split,               # percentage of data for validation
                           )
    
    # TODO:
    # create model
    my_model = model.Model(road_lambda      = parser.road_lambda,        # relative weight of road map loss
                           box_lambda       = parser.box_lambda,         # relative weight of bounding box loss
                           preload_backbone = parser.preload,            # whether to load pretrained weights
                           backbone_weights = parser.preload_weights,    # pretrained backbone weights if needed
                           )
    
    # create learner
    trainer = learner.Learner(access_mode      = parser.access_mode,     # os access mode for created files
                              experiment_name  = parser.experiment,      # name of experiment
                              model            = my_model,               # model
                              device           = device,                 # device to run experiment
                              myio             = data_handler,           # myio.IO object for loading data
                              save_dir         = parser.save_dir,        # directory to save results
                              max_steps        = parser.training_steps,  # maximum number of update steps
                              best_int         = parser.save_steps,      # interval for checking weights
                              verbose_int      = parser.verbose_steps,   # interval for logging information
                              max_grad_norm    = parser.max_grad_norm,   # maximum gradients to avoid exploding gradients
                              optimizer        = None,                   # optimizer for training
                              weight_decay     = parser.weight_decay,    # weight decay if using
                              lr               = parser.learning_rate,   # learning rate
                              eps              = parser.adam_epsilon,    # epsilon to use for adam
                              accumulate_int   = parser.accumulate_int,  # number of steps to accumulate gradients before stepping
                              batch_size       = parser.batch_size,      # batch size
                              warmup_pct       = parser.pct_start,       # percent of updates used to warm-up learning rate
                              save             = not parser.no_save,     # whether to save weights
                              patience         = parser.patience,        # number of checks without improvement before early stop
                              )

    # train model
    results = trainer.train(labeled = not parser.no_label, debug = parser.debug)

    results["experiment"] = parser.experiment
    
    # write results to "results.jsonl"
    results_name = os.path.join(parser.save_dir, "results.jsonl")
    with open(results_name, 'a') as f:
        f.write(json.dumps(results) + "\n")
    os.chmod(results_name, parser.access_mode)
    
    log.info("Results written to: {}".format(results_name))
    
    log.info("Total time is: {} min : {} sec".format((time.time()-start)//60, (time.time()-start)%60))
    
if __name__ == "__main__":
    main()
