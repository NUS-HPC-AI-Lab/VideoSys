python scripts/opensora/sample_opensora.py \
    --model_time_scale 1 \
    --model_space_scale 1 \
    --image_size 512 512 \
    --num_frames 16 \
    --fps 8 \
    --dtype fp16

# recommend setting
# python scripts/opensora/sample_opensora.py \
#     --model_time_scale 1 \
#     --model_space_scale 1 \
#     --image_size 512 512 \
#     --num_frames 16 \
#     --fps 8 \
#     --dtype fp16 \
#     --enable_flashattn \
#     --enable_layernorm_kernel \
#     --model_pretrained_path ckpt
