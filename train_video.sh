torchrun --standalone --nproc_per_node=2 train_video.py \
    --model DiT-XL/2 \
    --data_path ./processed \
    --batch_size 2 \
    # --enable_modulate_kernel \
    # --enable_layernorm_kernel \
    # --enable_flashattn \
    # --sequence_parallel_size 2
