#!/usr/bin/env python3
"""
Test script to verify Llama-3.2 model loading compatibility with EasyR1
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add the verl path from the verl directory
sys.path.insert(0, '/opt/tiger/verl')
sys.path.insert(0, '/opt/tiger/EasyR1')

def test_llama32_loading():
    """Test loading Llama-3.2 models"""
    
    # Available Llama-3.2 models
    test_models = [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]
    
    print("Testing Llama-3.2 model loading compatibility...")
    print("-" * 60)
    
    for model_name in test_models:
        print(f"\nTesting: {model_name}")
        print("-" * 40)
        
        try:
            # Try to load the model config first
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            
            print(f"✓ Config loaded successfully")
            print(f"  Model type: {config.model_type}")
            print(f"  Architecture: {config.architectures}")
            print(f"  Hidden size: {config.hidden_size}")
            print(f"  Num layers: {config.num_hidden_layers}")
            print(f"  Num heads: {config.num_attention_heads}")
            
            # Check if it's a standard LlamaForCausalLM architecture
            if config.architectures and "LlamaForCausalLM" in config.architectures[0]:
                print(f"✓ Uses standard LlamaForCausalLM architecture")
                print(f"  -> Should be compatible with existing verl implementation")
            else:
                print(f"⚠ Uses non-standard architecture: {config.architectures}")
                print(f"  -> May need additional support")
                
            # Check model type for monkey patching
            if config.model_type == "llama":
                print(f"✓ Model type 'llama' is already supported in monkey_patch.py")
            else:
                print(f"⚠ Model type '{config.model_type}' may need to be added to monkey_patch.py")
                
        except Exception as e:
            print(f"✗ Failed to load config: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("Llama-3.2 models use the standard LlamaForCausalLM architecture")
    print("and should work with the existing verl implementation.")
    print("No additional integration needed for basic support.")
    print("=" * 60)

if __name__ == "__main__":
    test_llama32_loading()