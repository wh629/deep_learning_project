module purge
module load anaconda3/5.3.1
module load cuda/10.1.105
module load gcc/6.3.0

source activate DL

EXPERIMENT=pretraining_1
PROJECT=/scratch/wh629/dl/project           # set to your project directory
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