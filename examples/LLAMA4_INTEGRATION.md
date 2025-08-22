# Llama4 Integration for EasyR1

This document describes the Llama4 integration implemented for the EasyR1 framework.

## Overview

Llama4 support has been added to EasyR1, enabling training of Meta's latest Mixture-of-Experts (MoE) models using the GRPO algorithm. The integration leverages HuggingFace Transformers' native Llama4 support.

## Supported Models

- **meta-llama/Llama-4-Scout-17B-16E-Instruct** (17B active params, 16 experts)
- **meta-llama/Llama-4-Scout-17B-16E** (Base model)
- **meta-llama/Llama-4-Maverick-17B-128E** (17B active params, 128 experts)

## Changes Made

### 1. Monkey Patch Support (`verl/models/monkey_patch.py`)
Added "llama4" to the supported model types for Ulysses sequence parallelism:

```python
if model_type in ("llama", "llama4", "gemma", "gemma2", "mistral", "qwen2", "qwen3", "qwen3_moe"):
    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
```

### 2. Training Script (`examples/llama4_critic_r1.sh`)
Updated the script to use:
- Correct Llama4 model path: `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- Optimized config file: `examples/llama4_config.yaml`
- Appropriate batch sizes for MoE architecture

### 3. Configuration (`examples/llama4_config.yaml`)
Created MoE-optimized configuration with:
- Reduced batch sizes for memory efficiency

## Usage

### Quick Start

```bash
cd /opt/tiger/EasyR1
bash examples/llama4_critic_r1.sh
```

### Custom Training

```bash
python3 -m verl.trainer.main \
    config=examples/llama4_config.yaml \
    worker.actor.model.model_path=meta-llama/Llama-4-Scout-17B-16E-Instruct \
    trainer.experiment_name=my_llama4_experiment
```

### Testing Model Loading

```bash
cd /opt/tiger/EasyR1
python examples/test_llama4_load.py
```