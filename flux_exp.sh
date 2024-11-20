#!/bin/bash

# 设置总数和每个GPU处理的数量
TOTAL_CAPTIONS=5000
CAPTIONS_PER_GPU=$((TOTAL_CAPTIONS / 5))

# 创建日志目录
mkdir -p ./logs

# 通用参数
COMMON_ARGS="--caption-file FID_caption.txt \
             --batch-size 4 \
             --height 1024 \
             --width 1024 \
             --guidance-scale 3.5 \
             --num-steps 50 \
             --max-seq-len 512 \
             --output-dir ./outputs/flux-pab"

# 为每个GPU分配任务
CUDA_VISIBLE_DEVICES=0 python examples/flux/sample_pab.py \
    --start-idx 0 \
    --end-idx $CAPTIONS_PER_GPU \
    --log-file ./logs/gpu0_generation.csv \
    $COMMON_ARGS \
    > ./logs/gpu0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python examples/flux/sample_pab.py \
    --start-idx $CAPTIONS_PER_GPU \
    --end-idx $((CAPTIONS_PER_GPU * 2)) \
    --log-file ./logs/gpu1_generation.csv \
    $COMMON_ARGS \
    > ./logs/gpu1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python examples/flux/sample_pab.py \
    --start-idx $((CAPTIONS_PER_GPU * 2)) \
    --end-idx $((CAPTIONS_PER_GPU * 3)) \
    --log-file ./logs/gpu2_generation.csv \
    $COMMON_ARGS \
    > ./logs/gpu2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python examples/flux/sample_pab.py \
    --start-idx $((CAPTIONS_PER_GPU * 3)) \
    --end-idx $((CAPTIONS_PER_GPU * 4)) \
    --log-file ./logs/gpu3_generation.csv \
    $COMMON_ARGS \
    > ./logs/gpu3.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python examples/flux/sample_pab.py \
    --start-idx $((CAPTIONS_PER_GPU * 4)) \
    --end-idx $TOTAL_CAPTIONS \
    --log-file ./logs/gpu4_generation.csv \
    $COMMON_ARGS \
    > ./logs/gpu4.log 2>&1 &

# 等待所有进程完成
wait