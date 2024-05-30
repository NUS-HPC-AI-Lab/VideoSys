BATCH_SIZE=1
LR=2e-5
DATA_PATH="./videos/demo.csv"
MODEL_PRETRAINED_PATH="pretrained/OpenSora-v1-HQ-16x512x512.pth"

torchrun --standalone --nproc_per_node=2 scripts/opensora/train_opensora_v1.py \
    --batch_size $BATCH_SIZE \
    --mixed_precision bf16 \
    --lr $LR \
    --grad_checkpoint \
    --data_path $DATA_PATH \
    --model_pretrained_path $MODEL_PRETRAINED_PATH

# recommend
# torchrun --standalone --nproc_per_node=8 scripts/opensora/train_opensora_v1.py \
#     --batch_size $BATCH_SIZE \
#     --mixed_precision bf16 \
#     --lr $LR \
#     --grad_checkpoint \
#     --data_path $DATA_PATH \
#     --enable_flashattn \
#     --enable_layernorm_kernel \
#     --text_speedup \
#     --model_pretrained_path $MODEL_PRETRAINED_PATH
