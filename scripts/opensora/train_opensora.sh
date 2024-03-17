BATCH_SIZE=2
LR=2e-5
DATA_PATH="/home/zhaowangbo/data_hdd/csv/inter4k_pexels_rp_fmin_48.csv"
MODEL_PRETRAINED_PATH="/home/zhaowangbo/zangwei/minisora_new/outputs/314-F16S3-PixArt-ST-XL-2/epoch128-global_step20000/ema.pt"

torchrun --standalone --nproc_per_node=2 train_opensora.py \
    --batch_size $BATCH_SIZE \
    --mixed_precision bf16 \
    --lr $LR \
    --grad_checkpoint \
    --data_path $DATA_PATH \
    --model_pretrained_path $MODEL_PRETRAINED_PATH

# recommend
# torchrun --standalone --nproc_per_node=2 train_opensora.py \
#     --batch_size $BATCH_SIZE \
#     --mixed_precision bf16 \
#     --lr $LR \
#     --grad_checkpoint \
#     --data_path $DATA_PATH \
#     --enable_flashattn \
#     --enable_layernorm_kernel \
#     --text_speedup \
#     --model_pretrained_path $MODEL_PRETRAINED_PATH
