BALANCED_DIR='./audioset/mp3/balanced_train_segments_mp3.hdf'
UNBALANCED_DIR='./audioset/mp3/unbalanced_train_segments_mp3.hdf'
EVAL_DIR='./audioset/mp3/eval_segments_mp3.hdf'
OTHER_DIR=''
OUTPUT_DIR='./audioset/output_Pretrain_submitit'
LOG_DIR='./audioset/log_Pretrain_submitit'

python3 trainer/submitit_pretrain.py \
    --nodes 8 \
    --batch_size 64 \
    --model mae_vit_base_patch16_dec512d8b \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --balanced_train_hdf5 ${BALANCED_DIR} \
    --unbalanced_train_hdf5 ${UNBALANCED_DIR} \
    --eval_hdf5 ${EVAL_DIR} \
    --other_hdf5_path ${OTHER_DIR} \
    --data_path ${IMAGENET_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --use_othersets
