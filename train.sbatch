#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=v100_sxm2_4,p40_4,p100_4,v100_pci_2
#SBATCH --time=48:00:00
#SBATCH --mem=32000
#SBATCH --job-name=team12_train

module purge
module load anaconda3/5.3.1
module load cuda/10.1.105
module load gcc/6.3.0

source activate DL

EXPERIMENT=train

python ./src/main.py \
      --experiment False_lr_0.005_bs_4_max-steps_5000_seed_403984 \
      --batch_size 1 \
      --accumulate_int 4 \
      --learning_rate 0.005 \
      --training_steps 5000 \
      --save_steps 500 \
      --verbose_steps 100 \
      --road_lambda 1.0 \
      --box_lambda 640000.0
