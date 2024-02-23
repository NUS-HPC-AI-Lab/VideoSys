torchrun --standalone --nproc_per_node=4 train_img.py \
    --model DiT-XL/2 \
    --batch-size 2 \
    # --enable_modulate_kernel \
    # --enable_layernorm_kernel \
    # --enable_flashattn \
    # --sequence_parallel_size 2
