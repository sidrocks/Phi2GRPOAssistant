import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys

def main():
    print("Initializing Validation for Phi-2 GRPO Model...")

    # Configuration
    # Adjust this path if you want to test a specific checkpoint instead of the final model
    adapter_path = "./results/phi2-grpo-final" 
    
    # Check for CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA is available. Using GPU: {device_name}")
        device = "cuda"
    else:
        print("CUDA is NOT available. Using CPU.")
        device = "cpu"

    # Determine optimal compute dtype (matching training)
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        print("BF16 is supported. Using bfloat16.")
    else:
        compute_dtype = torch.float16
        print("BF16 not supported. Using float16.")

    # 1. Load Base Model matching training config (No Quantization for stability)
    model_name = "microsoft/phi-2"
    print(f"Loading base model {model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Gen config

    # 2. Load Adapters
    print(f"Loading LoRA adapters from {adapter_path}...")
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
    except Exception as e:
        print(f"Error loading adapters: {e}")
        print("Ensure you have run training and the output directory exists.")
        return

    model.eval()
    
    # 3. Validation Prompts
    test_prompts = [
        "Explain the theory of relativity to a 5 year old.",
        "Write a python function to calculate fibonacci numbers.",
        "What are the benefits of exercise?",
        "Tengo un problema con mi coche, hace un ruido extraño.", # Test multilingual capability
    ]
    
    print("\n--- Starting Generation ---\n")
    
    for i, prompt in enumerate(test_prompts):
        print(f"Prompt {i+1}: {prompt}")
        print("-" * 20)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Calculate just the new tokens (optional, but cleaner)
        # simple decode includes prompt usually
        
        print(f"Response:\n{response}")
        print("=" * 40 + "\n")

if __name__ == "__main__":
    main()
