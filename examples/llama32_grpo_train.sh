#!/bin/bash

set -x

# export XDG_CACHE_HOME=/fsx-project/xywang96/.cache
# # (optional) explicitly set Triton's cache location
# export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
# # (optional) PyTorch-Inductor tuning cache
# export TORCHINDUCTOR_TUNING_DIR=$XDG_CACHE_HOME/torch/inductor

# # make sure the dirs exist
# mkdir -p $XDG_CACHE_HOME $TRITON_CACHE_DIR $TORCHINDUCTOR_TUNING_DIR

export PYTHONUNBUFFERED=1
export VLLM_USE_TRITON_FLASH_ATTN=0
# Model selection - choose between 1B and 3B variants
MODEL_PATH=/opt/tiger/checkpoints/Llama-3.2-1B-Instruct
# MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct

python3 -m verl.trainer.main \
    config=examples/llama32_config.yaml \
    worker.actor.model.model_path=${MODEL_PATH}