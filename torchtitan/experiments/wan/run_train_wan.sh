set -ex

CURDIR=$(cd $(dirname $0); pwd)

CONFIG_FILE=${CURDIR}/train_configs/wan2.1_t2v_debug.toml

CUDA_VISIBLE_DEVICES=4,5,6,7
NNODES=1
NPROC_PER_NODE=1
MASTER_ADDR="localhost"
MASTER_PORT="23500"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend=c10d \
    --local_ranks_filter=0 --role=rank --tee=3 \
    -m torchtitan.experiments.wan.train \
    --job.config_file ${CONFIG_FILE}