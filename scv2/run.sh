DATA_PATH_TRAIN='./scv2/data/scv2_train_mp3.hdf'
DATA_PATH_VAL='./scv2/data/scv2_validation_mp3.hdf'
DATA_PATH_TEST='./scv2/data/scv2_testing_mp3.hdf'
OUTPUT_DIR='./scv2/output_dir'
LOG_DIR='./scv2/log_dir'
FINETUNE='./checkpoint-75.pth'
MODEL='vit_base_patch16'
MODEL_TYPE='vit'
NORM_FILE='./scv2/mean_std_128.npy'

python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --use_env scv2/run.py \
    --model ${MODEL} \
    --model_type ${MODEL_TYPE} \
    --batch_size 64 \
    --data_path_train ${DATA_PATH_TRAIN} \
    --data_path_val ${DATA_PATH_VAL} \
    --data_path_test ${DATA_PATH_TEST} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --finetune ${FINETUNE} \
    --norm_file ${NORM_FILE}
