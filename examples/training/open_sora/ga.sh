#!/bin/bash
#PBS -P H100008
#PBS -l select=1:ngpus=8
#PBS -l place=vscatter
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -o ga.log

# =============== env params ================
# This script is for NSCC which uses PBS Pro as the scheduler

#-#PBS -P 11003730-#
# where the singularity image is saved
SCRATCH_PATH=$HPCTMP

# $TMPDIR set by PBS will intervene triton compilation inside singularity
export RECORD=$TMPDIR
unset TMPDIR
echo "JOB TMPDIR: $TMPDIR, record tmpdir: $RECORD"

cd $PBS_O_WORKDIR
echo "JOB ID: $PBS_JOBID, pwd: $PWD, pbs workdir: $PBS_O_WORKDIR"

# for torch.distributed
export NNODES=1
# export NODE_RANK=0
export GPUS_PER_NODE=8
export WORLD_SIZE=$(($NNODES*$GPUS_PER_NODE))
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE | awk -F'.' '{print $1}')
export MASTER_PORT=9529
echo "master node: $MASTER_ADDR"

# used by OpenMPI
export HOSTFILE="$PBS_JOBID.hostfile"
cat $PBS_NODEFILE | awk -F'.' '{for(i=1;i<=NF;i+=6) print $1 " slots="ENVIRON["GPUS_PER_NODE"]}' > $HOSTFILE
echo "detected hosts: $(cat $HOSTFILE)"

# refer to: https://apptainer.org/user-docs/master/gpu.html
# for apptainer, replace SINGULARITYENV_* with APPTAINERENV_*
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$(printf "%s," $(seq 0 $(($GPUS_PER_NODE-1))) | sed 's/,$//')
echo "singularity cuda visible devices: $SINGULARITYENV_CUDA_VISIBLE_DEVICES"

# =============== program params ================
export PYTHONPATH=$PYTHONPATH:$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# =============== launch cmd ================
# dynamic sp + ckpt + ga
# profile
# mpirun --hostfile $HOSTFILE --np $WORLD_SIZE -N $GPUS_PER_NODE --oversubscribe \
#     singularity exec --bind $SCRATCH_PATH:$SCRATCH_PATH --nv $SCRATCH_PATH/images/opensora_24.07v5.sif \
#     python scripts/opensora/train.py configs/opensora/train.yaml \
#     --preprocessed-data \
#     --dynamic-sp --dynamic-recompute --auto-grad-accumulation \
#     --profile --epochs 0 --outputs ga

# train
mpirun --hostfile $HOSTFILE --np $WORLD_SIZE -N $GPUS_PER_NODE --oversubscribe \
    singularity exec --bind $SCRATCH_PATH:$SCRATCH_PATH --nv $SCRATCH_PATH/images/opensora_24.07v5.sif \
    python scripts/opensora/train.py configs/opensora/train.yaml \
    --dummy-dataset --dummy-data-size 2000 --preprocessed-data --keep-last \
    --image-mixing-type exclusive --image-mixing-frac 5 \
    --dynamic-sp --dynamic-recompute --auto-grad-accumulation \
    --profile --epochs 2 --outputs ga \
    --profile-path archive/new_api/profile/ga/000-OpenSora/profile.json

rm $HOSTFILE
