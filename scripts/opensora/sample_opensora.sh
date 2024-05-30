# inference
torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_opensora.py \
    --image_size 240 426 \
    --num_frames 16 \
    --fps 24 \
    --dtype bf16 \
    --model_pretrained_path hpcai-tech/OpenSora-STDiT-v2-stage3

# recommend setting for speedup
#   --dtype bf16 \
#   --enable_flashattn \
#   --enable_layernorm_kernel \
#   --enable_t5_speedup \
