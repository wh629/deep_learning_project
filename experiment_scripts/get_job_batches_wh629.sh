module purge
module load anaconda3/5.3.1
module load cuda/10.1.105
module load gcc/6.3.0
source activate DL

PROJECT=/scratch/wh629/dl/project          # project directory
export DL_DATA_DIR=${PROJECT}/data         # data directory
export DL_RESULTS_DIR=${PROJECT}/results   # results directory
NETID=wh629                                # netid
TRIALS=1                                   # number of experiment to run
CAPACITY=2                                 # number of data batches a GPU can handle
CHECK=500                                  # number of update iterations between evaluations
LOG=100                                    # number of update iterations between logging information
ROAD=1.0                                   # relative weight of road map loss
BOX=1.0                                    # relative weight of box loss

python hyper_parameter_tuning.py \
	--user ${NETID} \
	--n-trials ${TRIALS} \
	--gpu-capacity ${CAPACITY} \
	--check_int ${CHECK} \
	--log_int ${LOG} \
	--road_lambda ${ROAD} \
	--box_lambda ${BOX} \
	--accumulate