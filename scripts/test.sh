BALANCED_DIR='./audioset/mp3/balanced_train_segments_mp3.hdf'
UNBALANCED_DIR='./audioset/mp3/unbalanced_train_segments_mp3.hdf'
EVAL_DIR='./audioset/mp3/eval_segments_mp3.hdf'
OUTPUT_DIR='./audioset/output_Finetune_vit'
LOG_DIR='./audioset/log_Finetune_vit'
RESUME='./AudioSet_Pretrained_Finetuned.pth'
MODEL='vit_base_patch16'
MODEL_TYPE='vit'
NORM_FILE='./audioset/mean_std_128.npy'
TEST_MODE='single'
CSV_FILE='./audioset/metadata/class_labels_indices.csv'
TEST_FILE='./audioset/audios/eval_segments/Y__p-iA312kg.wav'

python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --use_env trainer/test.py \
    --model ${MODEL} \
    --model_type ${MODEL_TYPE} \
    --test_mode ${TEST_MODE} \
    --test_file ${TEST_FILE} \
    --csv_file ${CSV_FILE} \
    --batch_size 128 \
    --balanced_train_hdf5 ${BALANCED_DIR} \
    --unbalanced_train_hdf5 ${UNBALANCED_DIR} \
    --eval_hdf5 ${EVAL_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --resume ${RESUME} \
    --norm_file ${NORM_FILE}
