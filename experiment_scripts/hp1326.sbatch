#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=v100_sxm2_4,p40_4,p100_4,v100_pci_2
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=hp1326@nyu.edu
#SBATCH --mem=32000
#SBATCH --job-name=dl
#SBATCH --output=/scratch/hp1326/%j.out

# customize netid for lines 9, 12
# also customize line 23

module purge
module load anaconda3/5.3.1
module load cuda/10.1.105
module load gcc/6.3.0

source activate DL

PROJECT=/scratch/hp1326           # set to your project directory
export DL_DATA_DIR=${PROJECT}/data
export DL_RESULTS_DIR=${PROJECT}/results

echo python $COMMAND
python $COMMAND
