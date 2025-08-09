#!/bin/bash
#PBS -P CFP02-CF-004
#PBS -l select=1:ngpus=8
#PBS -l place=vscatter
#PBS -l walltime=96:00:00
#PBS -j oe
#PBS -o 1x8-cogvideox-dcp-inter.log

# =============== env params ================
# This script is for NSCC which uses PBS Pro as the scheduler

# where the singularity image is saved
SCRATCH_PATH=$HPCTMP

cd $PBS_O_WORKDIR
echo "JOB ID: $PBS_JOBID, pwd: $PWD, pbs workdir: $PBS_O_WORKDIR"

# for torch.distributed
export NNODES=1
# export NODE_RANK=0
export GPUS_PER_NODE=8
export WORLD_SIZE=$(($NNODES*$GPUS_PER_NODE))
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE | awk -F'.' '{print $1}')
export MASTER_PORT=29502
echo "master node: $MASTER_ADDR"

# used by OpenMPI
export HOSTFILE="$PBS_JOBID.hostfile"
cat $PBS_NODEFILE | awk -F'.' '{for(i=1;i<=NF;i+=6) print $1 " slots="ENVIRON["GPUS_PER_NODE"]}' > $HOSTFILE
echo "detected hosts: $(cat $HOSTFILE)"

# refer to: https://apptainer.org/user-docs/master/gpu.html
# for apptainer, replace SINGULARITYENV_* with APPTAINERENV_*
# export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$(printf "%s," $(seq 0 $(($GPUS_PER_NODE-1))) | sed 's/,$//')
# echo "singularity cuda visible devices: $SINGULARITYENV_CUDA_VISIBLE_DEVICES"

# =============== program params ================
export PYTHONPATH=$PYTHONPATH:$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# =============== zipf-1 ================
mpirun --hostfile $HOSTFILE --np $WORLD_SIZE -N $GPUS_PER_NODE --oversubscribe \
    singularity exec --nv /app1/common/singularity-img/hopper/cuda/cuda_12.1.1-cudnn8-devel-ubuntu22.04.sif \
    /bin/bash -c "source /hpctmp/e1154485/venvs/videosys/bin/activate && \
    python examples/training/cogvideox/train.py \
    examples/training/cogvideox/configs/benchmarks/dcp_inter.yaml \
    --image-mixing-frac 1
"

# =============== zipf-10 ================
mpirun --hostfile $HOSTFILE --np $WORLD_SIZE -N $GPUS_PER_NODE --oversubscribe \
    singularity exec --nv /app1/common/singularity-img/hopper/cuda/cuda_12.1.1-cudnn8-devel-ubuntu22.04.sif \
    /bin/bash -c "source /hpctmp/e1154485/venvs/videosys/bin/activate && \
    python examples/training/cogvideox/train.py \
    examples/training/cogvideox/configs/benchmarks/dcp_inter.yaml \
    --image-mixing-frac 10
"

# =============== zipf-50 ================
mpirun --hostfile $HOSTFILE --np $WORLD_SIZE -N $GPUS_PER_NODE --oversubscribe \
    singularity exec --nv /app1/common/singularity-img/hopper/cuda/cuda_12.1.1-cudnn8-devel-ubuntu22.04.sif \
    /bin/bash -c "source /hpctmp/e1154485/venvs/videosys/bin/activate && \
    python examples/training/cogvideox/train.py \
    examples/training/cogvideox/configs/benchmarks/dcp_inter.yaml \
    --image-mixing-frac 50
"

rm $HOSTFILE
