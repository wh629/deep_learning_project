"""
Module for argument parcer.
"""
import argparse
import os

args = argparse.ArgumentParser(description='Deep Learning Competition')



args.add_argument('--experiment',
                  type=str,
                  default='testing',
                  help='name of experiment')
args.add_argument('--save_dir',
                  type=str, 
                  default='results',
                  help='directory to save results')
args.add_argument('--seed',
                  type=int,
                  default=42,
                  help='random seed')
args.add_argument('--run_log',
                  type=str,
                  default=os.path.join(os.getcwd(),'log'),
                  help='where to print run log')
args.add_argument('--access_mode',
                  type=int,
                  default=0o777,
                  help='access mode of files created')

# =============================================================================
# for dataloading
# =============================================================================
args.add_argument('--overwrite_cache', 
                  action='store_true',
                  help='overwrite the cached data sets')
args.add_argument('--doc_stride',
                  type=int,
                  default=128,
                  help='when chunking, how much stride between chunks')
args.add_argument('--data_dir',
                  type=str,
                  default='data',
                  help='directory storing all data')
args.add_argument('--batch_size', 
                  type=int, 
                  default=8,
                  help='batch size')
args.add_argument('--accumulate_int',
                  type=int,
                  default=1,
                  help='number of steps to accumulate gradient')

# =============================================================================
# for training
# =============================================================================
args.add_argument('--training_steps', 
                  type=int, 
                  default=100000,
                  help='number of updates for fine-tuning')
args.add_argument('--learning_rate', 
                  type=float, 
                  default=1e-4,
                  help='initial learning rate for Adam')
args.add_argument("--weight_decay",
                  type=float,
                  default=0.0,
                  help='weight decay if applied')
args.add_argument('--adam_epsilon',
                  type=float,
                  default=1e-8,
                  help='epsilon for Adam optimizer')
args.add_argument('--max_grad_norm',
                  type=float,
                  default=1.0,
                  help='max gradient norm for clipping')
args.add_argument('--logging_steps',
                  type=int,
                  default=1e4,
                  help='logs best weights every X update steps for experiment')
args.add_argument('--save_steps',
                  type=int,
                  default=500,
                  help='save best weights every X update steps')
args.add_argument('--verbose_steps',
                  type=int,
                  default=1000,
                  help='Log results ever X update steps')
args.add_argument('--pct_start',
                  type=float,
                  default=0.0,
                  help='percentage of cycle to warm-up learning rate')

def check_args(parser):
    """
    make sure directories exist
    """
    assert os.path.exists(parser.data_dir), "Data directory does not exist"
    assert os.path.exists(parser.save_dir), "Save directory does not exist"
    assert os.path.exists(parser.run_log),  "Run logging directory does not exist"