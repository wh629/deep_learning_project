module purge
module load anaconda3/5.3.1
module load cuda/10.1.105
module load gcc/6.3.0
source activate DL

PROJECT=/scratch/hp1326                    # project directory
export DL_DATA_DIR=${PROJECT}/data         # data directory
export DL_RESULTS_DIR=${PROJECT}/results   # results directory
NETID=hp1326                               # netid
TRIALS=10                                  # number of experiment to run
CAPACITY=1                                 # number of data batches a GPU can handle
CHECK=500                                  # number of update iterations between evaluations
LOG=100                                    # number of update iterations between logging information
ROAD=1.0                                   # relative weight of road map loss
BOX=10.0                                    # relative weight of box loss
PRETRAIN_WEIGHTS=''                        # absolute file name of preloaded weights with <path>\<filename>.pt

python hyper_parameter_tuning.py \
	--user ${NETID} \
	--n-trials ${TRIALS} \
	--gpu-capacity ${CAPACITY} \
	--check_int ${CHECK} \
	--log_int ${LOG} \
	--road_lambda ${ROAD} \
	--box_lambda ${BOX} \
	--accumulate #\                         <-- uncomment for loading pretrained weights
	#--preload \                            <-- uncomment for loading pretrained weights
	#--preload_weights ${PRETRAIN_WEIGHTS}  <-- uncomment for loading pretrained weights
