export PYTHONPATH=$PYTHONPATH:$PWD
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

singularity exec --bind $SCRATCH:$SCRATCH --nv $SCRATCH/images/opensora_24.09v1.sif \
    torchrun --standalone --nproc_per_node 4 scripts/opensora/train_ds.py configs/opensora/train.yaml \
    --dummy-dataset --dummy-data-size 2000 --dynamic-sp --profile --outputs fast
