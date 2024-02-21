torchrun --standalone --nproc_per_node=2 train_video.py \
    --model DiT-XL/2 \
    --data-path ./processed \
    --batch-size 1 \
    # --enable_modulate_kernel \
    # --enable_layernorm_kernel \
    # --enable_flashattn \
    # --sequence_parallel_size 2
