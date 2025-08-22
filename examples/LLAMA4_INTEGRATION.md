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
- Lower learning rate (5e-7) for large model stability
- Longer warmup ratio (0.1) for MoE training stability
- Conservative GPU memory utilization (0.5)
- Parameter and optimizer offloading enabled

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

## Architecture Considerations

### Mixture-of-Experts (MoE)
- Llama4 uses MoE architecture with sparse activation
- Only 17B parameters active during forward pass
- Requires careful memory management and batch size tuning

### Memory Requirements
- Recommended: 8x H100 GPUs (80GB each)
- Minimum: 4x H100 GPUs with parameter offloading
- Uses FSDP for model sharding across GPUs

### Performance Optimizations
- Flash Attention 2 support
- Gradient checkpointing enabled
- Parameter and optimizer offloading
- Dynamic batching for efficient training

## Configuration Parameters

Key parameters optimized for Llama4:

```yaml
worker:
  actor:
    global_batch_size: 64              # Reduced for MoE memory
    micro_batch_size_per_device_for_update: 2  # Conservative for 17B active
    model:
      enable_gradient_checkpointing: true
    optim:
      lr: 5.0e-7                       # Lower LR for stability
      lr_warmup_ratio: 0.1             # Longer warmup
    offload:
      offload_params: true             # Important for MoE
      offload_optimizer: true
  rollout:
    gpu_memory_utilization: 0.5        # Conservative for MoE
    tensor_parallel_size: 2            # TP for inference
```

## Known Limitations

1. **Model Availability**: Requires HuggingFace Transformers >= 4.51.0
2. **Memory Requirements**: Large memory footprint due to MoE architecture
3. **License**: Custom Llama 4 Community License (check Meta's terms)

## Troubleshooting

### OOM Errors
- Reduce `global_batch_size` to 32 or 16
- Increase `tensor_parallel_size` to 4 or 8
- Enable CPU offloading: `enable_cpu_offload: true`

### Slow Training
- Increase `micro_batch_size_per_device_for_update` if memory allows
- Disable gradient checkpointing for speed (at cost of memory)
- Reduce sequence length if possible

### Model Loading Issues
- Ensure transformers version >= 4.51.0
- Check HuggingFace authentication for gated models
- Verify sufficient disk space for model download

## Integration Details

The integration works by:

1. **Automatic Model Detection**: EasyR1 uses `AutoModelForCausalLM.from_pretrained()` which automatically detects Llama4 models
2. **Native Transformers Support**: Leverages HuggingFace's built-in Llama4ForCausalLM class
3. **Monkey Patching**: Applies Ulysses sequence parallelism optimizations
4. **FSDP Integration**: Uses PyTorch FSDP for distributed training

No changes to core EasyR1 logic were needed - the integration works through configuration and monkey patching.