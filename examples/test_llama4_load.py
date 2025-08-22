#!/usr/bin/env python3
"""
Simple test script to verify Llama4 model loading works with EasyR1
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def test_llama4_loading():
    """Test if we can load Llama4 model through transformers"""
    model_path = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    
    print(f"Testing Llama4 model loading: {model_path}")
    
    try:
        # Test config loading
        print("Loading config...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        print(f"Model type: {config.model_type}")
        print(f"Architecture: {config.architectures}")
        print(f"Hidden size: {config.hidden_size}")
        print(f"Num attention heads: {config.num_attention_heads}")
        print(f"Num layers: {config.num_hidden_layers}")
        
        # Test tokenizer loading
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        print(f"Tokenizer type: {type(tokenizer).__name__}")
        print(f"Vocab size: {tokenizer.vocab_size}")
        
        # Test model loading (CPU only for safety)
        print("\nLoading model on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        print(f"Model type: {type(model).__name__}")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Test a simple forward pass
        print("\nTesting tokenization and model structure...")
        test_text = "What is 2+2?"
        inputs = tokenizer(test_text, return_tensors="pt")
        print(f"Input shape: {inputs['input_ids'].shape}")
        
        print("✅ Llama4 loading test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Llama4 loading test FAILED: {e}")
        return False

if __name__ == "__main__":
    test_llama4_loading()