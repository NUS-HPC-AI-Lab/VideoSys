# data parallel inference
torchrun --standalone --nproc_per_node=2 scripts/opensora/sample_opensora.py \
    --model_time_scale 1 \
    --model_space_scale 1 \
    --image_size 512 512 \
    --num_frames 16 \
    --fps 8 \
    --dtype fp16 \
    --model_pretrained_path hpcai-tech/OpenSora-STDiT-v2-stage3

# sequence parallel (DSP) infernece
# torchrun --standalone --nproc_per_node=2 scripts/opensora/sample_opensora.py \
#     --model_time_scale 1 \
#     --model_space_scale 1 \
#     --image_size 512 512 \
#     --num_frames 16 \
#     --fps 8 \
#     --dtype fp16 \
#     --sequence_parallel_size 2 \
#     --model_pretrained_path ckpt_path

# recommend setting for speedup
#   --dtype fp16 \
#   --enable_flashattn \
#   --enable_layernorm_kernel \
#   --text_speedup \
