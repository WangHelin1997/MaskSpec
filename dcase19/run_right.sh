DATA_PATH_TRAIN='./dcase19/data/dcase19_train_right_mp3.hdf'
DATA_PATH_TEST='./dcase19/data/dcase19_test_right_mp3.hdf'
OUTPUT_DIR='./dcase19/output_dir_right'
LOG_DIR='./dcase19/log_dir_right'
FINETUNE='./checkpoint-75.pth'
MODEL='vit_base_patch16'
MODEL_TYPE='vit'
NORM_FILE='./dcase19/mean_std_right_128.npy'

CUDA_VISIBLE_DEVICES=1 python3 dcase19/run.py \
    --model ${MODEL} \
    --model_type ${MODEL_TYPE} \
    --batch_size 64 \
    --data_path_train ${DATA_PATH_TRAIN} \
    --data_path_test ${DATA_PATH_TEST} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --finetune ${FINETUNE} \
    --norm_file ${NORM_FILE}
