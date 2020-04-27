#!/bin/bash
#
#SBATCH --job-name=myDLGPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=hp1326@nyu.edu

module load anaconda3/5.3.1
source activate /home/hp1326/miniconda3/envs/pDL

PYTHONPATH=./src python src/pretraining/pretrain.py --batch_size 16
