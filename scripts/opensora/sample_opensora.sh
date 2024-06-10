H=1024
W=1024
# H=512
# W=512

# inference
torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_opensora.py \
    --image_size $H $W \
    --num_frames 16 \
    --fps 24 \
    --dtype bf16 \
    --model_pretrained_path hpcai-tech/OpenSora-STDiT-v2-stage3 \
    --save_dir ./samples/output \
    --enable_flashattn \
    --enable_t5_speedup

# recommend setting for speedup
#   --dtype bf16 \
#   --enable_flashattn \
#   --enable_layernorm_kernel \
#   --enable_t5_speedup \
