#!/bin/bash

set -x

# export XDG_CACHE_HOME=/fsx-project/xywang96/.cache
# # (optional) explicitly set Triton’s cache location
# export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
# # (optional) PyTorch-Inductor tuning cache
# export TORCHINDUCTOR_TUNING_DIR=$XDG_CACHE_HOME/torch/inductor

# # make sure the dirs exist
# mkdir -p $XDG_CACHE_HOME $TRITON_CACHE_DIR $TORCHINDUCTOR_TUNING_DIR

export PYTHONUNBUFFERED=1

MODEL_PATH=meta-llama/Llama-4-Scout-17B-16E-Instruct

python3 -m verl.trainer.main \
    config=examples/llama4_config.yaml \
    worker.actor.model.model_path=${MODEL_PATH}