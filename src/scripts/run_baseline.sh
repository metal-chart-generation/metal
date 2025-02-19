DATASET_DIR={PATH_TO_DATASET_DIR} # e.g. ../../dataset
WORKING_HOME_DIR={WORKING_HOME_DIR} # path to your desired result and working directory
DATA_RANGE={YOUR_DATA_RANGE} # e.g. 0-100
MODEL={YOUR_MODEL} # supported model: llama3_2, gpt-4o

SYSTEM={YOUR_SYSTEM} # supported system: HintEnhanced, Best-of-N, SelfRevision

EXP_NAME=$MODEL\_$SYSTEM\_$DATA_RANGE
MAX_ITER={YOUR_MAX_ITER}
N_PROCESS=8
CUDA_DEVICES=0,1

AGENT_KWARGS="{\"cuda_device\": \"$CUDA_DEVICES\", \"system\": \"$SYSTEM\"}"

WORKING_DIR=$WORKING_HOME_DIR/$EXP_NAME/
LOG_FILE_PATH=$WORKING_DIR/$EXP_NAME.error

mkdir -p $WORKING_DIR

screen -dmS $EXP_NAME bash -c "cd $WORKING_DIR; python3 -u ../src/Baselines.py --exp_name $EXP_NAME --max_iter $MAX_ITER --model $MODEL --data_range $DATA_RANGE --dataset_dir $DATASET_DIR --working_dir $WORKING_DIR --n_process $N_PROCESS --agent_kwargs '$AGENT_KWARGS' 2> $LOG_FILE_PATH"