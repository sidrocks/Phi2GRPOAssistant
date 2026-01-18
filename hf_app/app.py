import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# --- Configuration ---
BASE_MODEL_NAME = "microsoft/phi-2"
# Path to your adapters. 
# For a real HF Space, you would typically upload the adapter files to the Space 
# and prompt the path relative to the root, e.g., "./adapter_model"
ADAPTER_PATH = "./model_adapters" 

print("Initializing Model...")

# Check for CUDA
if torch.cuda.is_available():
    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("Using CPU")

# Determine dtype
compute_dtype = torch.float32
if device == "cuda":
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        print("Using bfloat16")
    else:
        compute_dtype = torch.float16
        print("Using float16")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load Model
try:
    print(f"Loading base model {BASE_MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=compute_dtype,
        device_map=device,
        trust_remote_code=True
    )
    
    print(f"Check for adapters at {ADAPTER_PATH}")
    if os.path.exists(ADAPTER_PATH):
        print(f"Loading adapters from {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        print(f"WARNING: Adapter path {ADAPTER_PATH} not found. Using base model only.")
        model = base_model
        
    model.eval()
    print("Model loaded successfully.")
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

def generate_response(prompt, temperature, top_p, max_new_tokens):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Optional: Clean up response if it repeats the prompt (usually model does)
        # response = response[len(prompt):] 
        return response
        
    except Exception as e:
        return f"Error during generation: {str(e)}"

# --- Gradio UI ---
with gr.Blocks(title="Phi-2 GRPO Fine-tuned Model") as demo:
    gr.Markdown("# Phi-2 GRPO Assistant")
    gr.Markdown("Query the fine-tuned Phi-2 model.")
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Input Prompt", 
                lines=5, 
                placeholder="Enter your prompt here..."
            )
            
            with gr.Accordion("Hyperparameters", open=True):
                temp_slider = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.7, step=0.1, 
                    label="Temperature (Creativity)"
                )
                top_p_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.9, step=0.05, 
                    label="Top-p (Nucleus Sampling)"
                )
                max_tokens_slider = gr.Slider(
                    minimum=16, maximum=512, value=200, step=16, 
                    label="Max New Tokens"
                )
            
            submit_btn = gr.Button("Generate", variant="primary")
            
        with gr.Column(scale=1):
            output_box = gr.Textbox(label="Model Response", lines=10, interactive=False)
            
    # Sample Inputs
    examples = gr.Examples(
        examples=[
            ["Explain quantum computing to a high school student."],
            ["Write a Python function to quicksort a list."],
            ["What are the health benefits of meditation?"],
            ["Traduce 'Hello, how are you?' al español."],
            ["Create a story about a robot who wants to be a chef."]
        ],
        inputs=prompt_input
    )

    submit_btn.click(
        fn=generate_response,
        inputs=[prompt_input, temp_slider, top_p_slider, max_tokens_slider],
        outputs=output_box
    )

if __name__ == "__main__":
    demo.launch()
