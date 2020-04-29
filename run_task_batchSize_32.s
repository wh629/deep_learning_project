#!/bin/bash
#
#SBATCH --job-name=myDLGPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=v100_sxm2_4,p40_4,p100_4,v100_pci_2,k80_4
#SBATCH --mail-user=hp1326@nyu.edu

module purge
module load anaconda3/5.3.1
source activate DL

PYTHONPATH=./src python src/pretraining/pretrain.py --batch_size 32 --permutations_k 8;
PYTHONPATH=./src python src/pretraining/pretrain.py --batch_size 32 --permutations_k 16;
PYTHONPATH=./src python src/pretraining/pretrain.py --batch_size 32 --permutations_k 32;
