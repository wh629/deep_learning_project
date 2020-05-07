#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=v100_sxm2_4,p40_4,p100_4,v100_pci_2
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=hp1326@nyu.edu
#SBATCH --mem=64000
#SBATCH --job-name=dl_pre
#SBATCH --output=/scratch/hp1326/pretrain_%j.out


module purge
module load anaconda3/5.3.1
module load cuda/10.1.105
module load gcc/6.3.0

source activate DL

EXPERIMENT=pretraining_3_lr0.01_5epochs_hp1326
PROJECT=/scratch/hp1326/           # set to your project directory
export DL_DATA_DIR=${PROJECT}/data
export DL_RESULTS_DIR=${PROJECT}/results

python ./src/pretraining/pretrain.py \
      --batch_size 2 \
      --permutations_k 35 \
      --num_epochs 5 \
      --accum_grad 4 \
      --experiment ${EXPERIMENT} \
      --log_steps 100 \
      --save_steps 500 \
      --patience 5 \
      --lr 0.01 \
      --split 0.1
