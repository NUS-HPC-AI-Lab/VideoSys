NUM_GPUS=1
torchrun --standalone --nproc_per_node=$NUM_GPUS scripts/cogvideo/sample.py
