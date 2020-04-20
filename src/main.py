"""
Main run script to execute experiments and analysis

TO DO: Meta-Learning (OML)
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
    
    data_handler = myio.IO(parser.data_dir,
                           batch_size = parser.batch_size,
                           shuffle=True,
                           )
    
    # create model
    my_model = model.Model(None)
    
    # create learner object for BERT model
    trainer = learner.Learner(parser.access_mode,
                              parser.experiment,
                              my_model,
                              device,
                              data_handler,
                              parser.save_dir,
                              max_steps = parser.training_steps,
                              log_int = parser.logging_steps,
                              best_int = parser.save_steps,
                              verbose_int = parser.verbose_steps,
                              max_grad_norm = parser.max_grad_norm,
                              optimizer = None,
                              weight_decay = parser.weight_decay,
                              lr = parser.learning_rate,
                              eps = parser.adam_epsilon,
                              accumulate_int = parser.accumulate_int
                              )
    results = trainer.train()
    results["Experiment"] = parser.experiment
    
    # write results to "val_results.jsonl"
    results_name = os.path.join(parser.save_dir, "val_results.jsonl")
    with open(results_name, 'a') as f:
        f.write(json.dumps(results) + "\n")
    os.chmod(results_name, parser.access_mode)
    log.info("Baseline results written to: {}".format(results_name))
    
    log.info("Total time is: {}min : {}s".format((time.time()-start)//60, (time.time()-start)%60))
    
if __name__ == "__main__":
    main()