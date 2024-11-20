#!/bin/bash
# run_parallel.sh

# 设置总数和每个GPU处理的数量
TOTAL_CAPTIONS=5000
CAPTIONS_PER_GPU=$((TOTAL_CAPTIONS / 5))

# 创建日志目录
mkdir -p ./logs

# 通用参数
COMMON_ARGS="--caption-file FID_caption.txt \
             --batch-size 8 \
             --height 1024 \
             --width 1024 \
             --guidance-scale 3.5 \
             --num-steps 50 \
             --max-seq-len 512 \
             --output-dir ./outputs/flux-pab"

# 为每个GPU分配任务
CUDA_VISIBLE_DEVICES=0 python process_captions.py --start-idx 0 --end-idx $CAPTIONS_PER_GPU \
    --log-file ./logs/gpu0_log.csv $COMMON_ARGS &

CUDA_VISIBLE_DEVICES=1 python process_captions.py --start-idx $CAPTIONS_PER_GPU --end-idx $((CAPTIONS_PER_GPU * 2)) \
    --log-file ./logs/gpu1_log.csv $COMMON_ARGS &

CUDA_VISIBLE_DEVICES=2 python process_captions.py --start-idx $((CAPTIONS_PER_GPU * 2)) --end-idx $((CAPTIONS_PER_GPU * 3)) \
    --log-file ./logs/gpu2_log.csv $COMMON_ARGS &

CUDA_VISIBLE_DEVICES=3 python process_captions.py --start-idx $((CAPTIONS_PER_GPU * 3)) --end-idx $((CAPTIONS_PER_GPU * 4)) \
    --log-file ./logs/gpu3_log.csv $COMMON_ARGS &

CUDA_VISIBLE_DEVICES=4 python process_captions.py --start-idx $((CAPTIONS_PER_GPU * 4)) --end-idx $TOTAL_CAPTIONS \
    --log-file ./logs/gpu4_log.csv $COMMON_ARGS &

# 等待所有进程完成
wait

echo "All processes completed!"