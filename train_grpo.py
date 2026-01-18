import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig
import os

def reward_len(completions, **kwargs):
    """
    Simple reward function that rewards longer completions.
    In a real scenario, this would be a sentiment model or task-specific verifier.
    """
    return [float(len(c)) for c in completions]

def main():
    print("Initializing GRPO Training for Phi-2...")
    
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For better debug trace
    
    # Check for CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA is available. Using GPU: {device_name}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        device = "cuda"
    else:
        print("CUDA is NOT available. Using CPU. Training will be extremely slow.")
        device = "cpu"

    # Determine optimal compute dtype and attention implementation
    attn_implementation = "eager"
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        print(f"BF16 is supported on {device_name}. Using bfloat16 to prevent overflow (NaNs).")
        # Try to use Flash Attention 2 if available (Ampere or newer)
        try:
            import flash_attn
            attn_implementation = "flash_attention_2"
            print("Flash Attention 2 is available and enabled.")
        except ImportError:
            print("Flash Attention 2 not installed. Using eager attention.")
    else:
        compute_dtype = torch.float16
        print("BF16 not supported. Using float16 (risk of overflow).")
        print("Flash Attention 2 requires BF16 (Ampere+). Using eager attention.")

    # 1. Load Dataset
    dataset_path = "./data/oasst1_prompts"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run prepare_data.py first.")
        
    dataset = load_from_disk(dataset_path)
    print(f"Loaded dataset with {len(dataset)} prompts.")

    # Filter dataset
    dataset = dataset.filter(lambda x: len(x['prompt']) < 2000) 
    print(f"Filtered dataset size: {len(dataset)}")

    # 2. Model & Tokenizer Config
    model_name = "microsoft/phi-2"
    
    # DEBUG: Disable 4-bit quantization to rule out BitsAndBytes instability (NaNs).
    # Phi-2 is small enough (2.7B) to fit in decent GPU memory (approx 6GB in bf16).
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=compute_dtype, 
    # )

    print(f"Loading model {model_name} in {compute_dtype} (No Quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config, # Disabled
        torch_dtype=compute_dtype, # Load directly in bf16/fp16
        device_map="auto", 
        trust_remote_code=True,
        attn_implementation=attn_implementation 
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Left padding is required for generation-based training
    tokenizer.padding_side = "left"
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model vocab size: {model.config.vocab_size}")
    if len(tokenizer) > model.config.vocab_size:
        print("WARNING: Tokenizer has more tokens than the model! This will cause CUDA asserts.")
        print("Resizing model embeddings to match tokenizer...")
        model.resize_token_embeddings(len(tokenizer))

    # CRITICAL FIXES for "device-side assert":
    # 1. Do NOT resize embeddings if we are just reusing EOS as PAD. 
    #    Resizing without adding new tokens can introduce uninitialized weights or vocab mismatches.
    # model.resize_token_embeddings(len(tokenizer)) <- REVERTED
    
    # 2. Explicitly update model config and generation config
    model.config.pad_token_id = tokenizer.pad_token_id
    if model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        # Ensure sampling is stable
        model.generation_config.do_sample = True
        model.generation_config.temperature = 0.7 # Add valid temperature

    # 3. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["Wqkv", "out_proj", "fc1", "fc2"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 4. GRPO Config
    import sys
    max_steps = 100
    if len(sys.argv) > 1:
        try:
            max_steps = int(sys.argv[1])
            print(f"Overriding max_steps to {max_steps}")
        except ValueError:
            pass

    # OPTIMIZATION NOTES:
    # 1. num_generations: Lowering this speeds up training (fewer generations per step). 
    #    GRPO needs at least a group, but 2-4 is common. We use 4 by default.
    # 2. max_completion_length: Generating long sequences is slow. 
    #    Reduced to 256 for speed. Increase if you need longer answers.
    training_args = GRPOConfig(
        output_dir="./results/phi2-grpo",
        learning_rate=1e-5,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,
        max_prompt_length=512,
        max_completion_length=256, # Kept at 256 as requested
        num_generations=2, # REDUCED from 4 to 2 for SPEED
        max_steps=max_steps,
        save_steps=50,
        logging_steps=10,
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none", 
        gradient_checkpointing=True, 
    )

    # 5. Initialize Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Training complete. Saving model...")
    trainer.save_model("./results/phi2-grpo-final")
    print("Model saved to ./results/phi2-grpo-final")

if __name__ == "__main__":
    main()
