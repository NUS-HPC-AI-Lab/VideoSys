export PYTHONPATH=$PYTHONPATH:$PWD

# NUM_GPUS=2
# torchrun --standalone --nproc_per_node=$NUM_GPUS scripts/opensora/sample.py

python scripts/opensora/sample.py
