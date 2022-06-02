DATA_PATH_TRAIN='./openmic18/data/mp3/openmic_train.csv_mp3.hdf'
DATA_PATH_TEST='./openmic18/data/mp3/openmic_test.csv_mp3.hdf'
OUTPUT_DIR='./openmic18/output_dir'
LOG_DIR='./openmic18/log_dir'
FINETUNE='./checkpoint-75.pth'
MODEL='vit_base_patch16'
MODEL_TYPE='vit'
NORM_FILE='./openmic18/mean_std_128.npy'

CUDA_VISIBLE_DEVICES=3 python3 openmic18/run.py \
    --model ${MODEL} \
    --model_type ${MODEL_TYPE} \
    --batch_size 64 \
    --data_path_train ${DATA_PATH_TRAIN} \
    --data_path_test ${DATA_PATH_TEST} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --finetune ${FINETUNE} \
    --norm_file ${NORM_FILE}
