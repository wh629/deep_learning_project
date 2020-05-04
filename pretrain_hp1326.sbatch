#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=hp1326@nyu.edu
#SBATCH --mem=32000
#SBATCH --job-name=dl
#SBATCH --output=/scratch/hp1326/%j.out


module purge
module load anaconda3/5.3.1
module load cuda/10.1.105
module load gcc/6.3.0

source activate DL

EXPERIMENT=pretraining_hp1326
PROJECT=/scratch/hp1326/           # set to your project directory
export DL_DATA_DIR=${PROJECT}/data
export DL_RESULTS_DIR=${PROJECT}/results

python ./src/pretraining/pretrain.py \
      --batch_size 2 \
      --permutations_k 35 \
      --num_epochs 2 \
      --accum_grad 4 \
      --experiment ${EXPERIMENT} \
      --log_steps 100 \
      --save_steps 500 \
      --patience 5 \
      --lr 0.1 \
      --split 0.1