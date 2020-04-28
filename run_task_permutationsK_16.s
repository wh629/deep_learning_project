#!/bin/bash
#
#SBATCH --job-name=myDLGPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=v100_sxm2_4,p40_4,p100_4,v100_pci_2,k80_4,k80_8
#SBATCH --mail-user=hp1326@nyu.edu

module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0
source activate /home/hp1326/miniconda3/envs/pDL

PYTHONPATH=./src python src/pretraining/pretrain.py --batch_size 64 --permutations_k 16;
PYTHONPATH=./src python src/pretraining/pretrain.py --batch_size 128 --permutations_k 16;