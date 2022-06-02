DATA_PATH_FOLD1='./esc50/data/esc50_mp3_fold1.hdf'
DATA_PATH_FOLD2='./esc50/data/esc50_mp3_fold2.hdf'
DATA_PATH_FOLD3='./esc50/data/esc50_mp3_fold3.hdf'
DATA_PATH_FOLD4='./esc50/data/esc50_mp3_fold4.hdf'
DATA_PATH_FOLD5='./esc50/data/esc50_mp3_fold5.hdf'
OUTPUT_DIR='./esc50/output_dir'
LOG_DIR='./esc50/log_dir'
FINETUNE='./checkpoint-75.pth'
MODEL='vit_base_patch16'
MODEL_TYPE='vit'
NORM_FILE='./audioset/mean_std_128.npy'

python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --use_env esc50/run.py \
    --model ${MODEL} \
    --model_type ${MODEL_TYPE} \
    --batch_size 64 \
    --data_path_fold1 ${DATA_PATH_FOLD1} \
    --data_path_fold2 ${DATA_PATH_FOLD2} \
    --data_path_fold3 ${DATA_PATH_FOLD3} \
    --data_path_fold4 ${DATA_PATH_FOLD4} \
    --data_path_fold5 ${DATA_PATH_FOLD5} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --finetune ${FINETUNE} \
    --norm_file ${NORM_FILE}
