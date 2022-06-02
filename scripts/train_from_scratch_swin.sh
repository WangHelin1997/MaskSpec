BALANCED_DIR='./audioset/mp3/balanced_train_segments_mp3.hdf'
UNBALANCED_DIR='./audioset/mp3/unbalanced_train_segments_mp3.hdf'
EVAL_DIR='./audioset/mp3/eval_segments_mp3.hdf'
OUTPUT_DIR='./audioset/output_TrainFromScratch_swin'
LOG_DIR='./audioset/log_TrainFromScratch_swin'
MODEL='swinTrans_base'
MODEL_TYPE='swin'
NORM_FILE='./audioset/mean_std_64.npy'

python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --use_env trainer/main_finetune.py \
    --model ${MODEL} \
    --model_type ${MODEL_TYPE} \
    --batch_size 128 \
    --balanced_train_hdf5 ${BALANCED_DIR} \
    --unbalanced_train_hdf5 ${UNBALANCED_DIR} \
    --eval_hdf5 ${EVAL_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --norm_file ${NORM_FILE}
