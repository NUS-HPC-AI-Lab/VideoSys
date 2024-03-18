torchrun --standalone --nproc_per_node=2 scripts/dit/train_dit.py \
    --model DiT-XL/2 \
    --batch_size 2 \
    --num_classes 10

# recommend setting
# torchrun --standalone --nproc_per_node=8 scripts/dit/train_dit.py \
#     --model DiT-XL/2 \
#     --batch_size 180 \
#     --enable_layernorm_kernel \
#     --enable_flashattn \
#     --mixed_precision bf16 \
#     --num_classes 1000
