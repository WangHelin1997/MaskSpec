DATA_PATH_TEST_LEFT='./dcase19/data/dcase19_test_left_mp3.hdf'
DATA_PATH_TEST_RIGHT='./dcase19/data/dcase19_test_right_mp3.hdf'
DATA_PATH_TEST_MID='./dcase19/data/dcase19_test_mid_mp3.hdf'
OUTPUT_DIR='./dcase19/output_dir_ensemble'
LOG_DIR='./dcase19/log_dir_ensemble'
FINETUNE='./checkpoint-75.pth'
MODEL='vit_base_patch16'
MODEL_TYPE='vit'
NORM_FILE_LEFT='./dcase19/mean_std_left_128.npy'
NORM_FILE_RIGHT='./dcase19/mean_std_right_128.npy'
NORM_FILE_MID='./dcase19/mean_std_mid_128.npy'
RESUME_LEFT='./dcase19/output_dir_left/checkpoint-195.pth'
RESUME_RIGHT='./dcase19/output_dir_right/checkpoint-195.pth'
RESUME_MID='./dcase19/output_dir_mid/checkpoint-195.pth'

CUDA_VISIBLE_DEVICES=0 python3 dcase19/run_ensemble.py \
    --model ${MODEL} \
    --model_type ${MODEL_TYPE} \
    --batch_size 64 \
    --data_path_test_left ${DATA_PATH_TEST_LEFT} \
    --data_path_test_right ${DATA_PATH_TEST_RIGHT} \
    --data_path_test_mid ${DATA_PATH_TEST_MID} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --finetune ${FINETUNE} \
    --norm_file_left ${NORM_FILE_LEFT} \
    --norm_file_right ${NORM_FILE_RIGHT} \
    --norm_file_mid ${NORM_FILE_MID} \
    --resume_left ${RESUME_LEFT} \
    --resume_right ${RESUME_RIGHT} \
    --resume_mid ${RESUME_MID}
