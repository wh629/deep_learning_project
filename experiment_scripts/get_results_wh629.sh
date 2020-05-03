module purge
module load anaconda3/5.3.1
module load cuda/10.1.105
module load gcc/6.3.0
source activate DL

PROJECT=/scratch/wh629/dl/project
export DL_DATA_DIR=${PROJECT}/data
export DL_RESULTS_DIR=${PROJECT}/results

python collect_results.py