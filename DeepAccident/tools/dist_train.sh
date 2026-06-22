#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

TORCH_DISTRIBUTED_DEBUG=DETAIL \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

# bash tools/dist_train.sh projects/configs/DeepAccident_tiny_v2x_4agents_infra.py 4 --load-from ./data/DeepAccident_data/beverse_tiny.pth --work-dir work_dirs/DeepAccident_tiny_v2x_4agents_infra
