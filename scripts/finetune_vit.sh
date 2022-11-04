BALANCED_DIR='/./audioset/mp3/balanced_train_segments_mp3.hdf'
UNBALANCED_DIR='./audioset/mp3/unbalanced_train_segments_mp3.hdf'
EVAL_DIR='./audioset/mp3/eval_segments_mp3.hdf'
OUTPUT_DIR='./audioset/output_Finetune_vit'
LOG_DIR='./audioset/log_Finetune_vit'
FINETUNE='./audioset/output_Pretrain_vit/checkpoint-60.pth'
MODEL='vit_base_patch16'
MODEL_TYPE='vit'
NORM_FILE='./audioset/mean_std_128.npy'

python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --use_env trainer/main_finetune.py \
    --model ${MODEL} \
    --model_type ${MODEL_TYPE} \
    --batch_size 64 \
    --seed 3407 \
    --balanced_train_hdf5 ${BALANCED_DIR} \
    --unbalanced_train_hdf5 ${UNBALANCED_DIR} \
    --eval_hdf5 ${EVAL_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --finetune ${FINETUNE} \
    --norm_file ${NORM_FILE}
