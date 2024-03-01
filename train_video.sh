torchrun --standalone --nproc_per_node=2 train.py \
    --model VDiT-XL/1x2x2 \
    --use_video \
    --data_path ./videos/demo.csv \
    --batch_size 1 \
    --num_frames 16 \
    --image_size 256 \
    --frame_interval 3
