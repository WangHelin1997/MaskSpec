BALANCED_DIR='./audioset/mp3/balanced_train_segments_mp3.hdf'
UNBALANCED_DIR='./audioset/mp3/unbalanced_train_segments_mp3.hdf'
EVAL_DIR='./audioset/mp3/eval_segments_mp3.hdf'
OUTPUT_DIR='./audioset/output_Pretrain_vit'
LOG_DIR='./audioset/log_Pretrain_vit'
MODEL='mae_vit_base_patch16_dec512d8b'
MODEL_TYPE='vit'
NORM_FILE='./audioset/mean_std_128.npy'

python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --use_env trainer/main_pretrain.py \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --model ${MODEL} \
    --model_type ${MODEL_TYPE} \
    --batch_size 64 \
    --balanced_train_hdf5 ${BALANCED_DIR} \
    --unbalanced_train_hdf5 ${UNBALANCED_DIR} \
    --eval_hdf5 ${EVAL_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --norm_file ${NORM_FILE}
