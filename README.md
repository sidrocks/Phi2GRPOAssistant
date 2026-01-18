# Phi-2 Fine-tuning with GRPO

This project implements the fine-tuning of Microsoft's **Phi-2** (2.7B parameter) language model using **GRPO (Group Relative Policy Optimization)** and **LoRA (Low-Rank Adaptation)** on the **OpenAssistant/oasst1** dataset.

## 🧠 Theory & Architecture

### 1. Model: Microsoft Phi-2
Phi-2 is a Transformer-based Small Language Model (SLM) with 2.7 billion parameters. It was trained on "textbook quality" data, making it highly efficient for reasoning capabilities despite its size.

### 2. Fine-tuning Strategy: LoRA
We use **LoRA** to fine-tune the model efficiently. Instead of updating all weights, we inject trainable low-rank decomposition matrices into existing layers (Target modules: `Wqkv`, `out_proj`, `fc1`, `fc2`).
- **Reduction**: Drastically reduces the number of trainable parameters.
- **Precision**: The base model is loaded in `bfloat16` (Brain Floating Point) to ensure stability and performance on Ampere+ GPUs (RTX 30 series/40 series/Ada), preventing floating-point overflows common in `float16`. 4-bit quantization was explored but disabled for this project to ensure maximum stability during the generation phases of RLHF.

### 3. Training Algorithm: GRPO
**Group Relative Policy Optimization (GRPO)** is an RLHF (Reinforcement Learning from Human Feedback) algorithm provided by Hugging Face's `trl` library.
- **Workflow**: For each prompt, the model generates multiple completions (a "group").
- **Reward**: A reward function evaluates these completions. (In this demo, a length-based heuristic is used, but this can be swapped for a true Reward Model).
- **Optimization**: The model is updated to maximize the likelihood of high-reward completions *relative* to the group average. This eliminates the need for a separate "Critic" or Value Network (unlike PPO), saving significant memory.

## 📂 Project Structure

- `prepare_data.py`: Loads `OpenAssistant/oasst1`, handles multi-language extraction, and filters prompts.
- `train_grpo.py`: Main training script. Configures the model, LoRA adapters, and runs the `GRPOTrainer`.
- `validate_model.py`: Inference script to test the model with your trained adapters.
- `inspect_data.py`: Utility to explore dataset samples.
- `hf_app/`: Contains the Gradio web application for Hugging Face Spaces.

## 🚀 Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Extract prompts from the OpenAssistant dataset:
```bash
python prepare_data.py
```
This saves the processed dataset to `./data/oasst1_prompts`.

### 3. Training
Start the GRPO training. 
- **Arguments**: You can pass the number of maximum steps (default 100).
- **Memory**: Requires a GPU with ~12-16GB VRAM for decent batch sizes (using `bfloat16`).
```bash
# Run for 1000 steps
python train_grpo.py 1000
```
The model checkpoints and final adapter will be saved to `./results/phi2-grpo-final`.

### 4. Validation
Test the trained model with sample prompts:
```bash
python validate_model.py
```

## 🌐 Hugging Face Gradio App

A web interface is provided to query the model interactively.

### Running Locally
1. **Copy Adapters**: Ensure your trained layout is correct. Copy the contents of `./results/phi2-grpo-final` into `./hf_app/model_adapters`.
   *(If you skip this, the app will just use the base Phi-2 model)*.
   
2. **Install App Deps**:
   ```bash
   cd hf_app
   pip install -r requirements.txt
   ```

3. **Launch**:
   ```bash
   python app.py
   ```
   Open the displayed URL (e.g., `http://127.0.0.1:7860`).

### Deployment
To deploy to Hugging Face Spaces:
1. Create a Space (SDK: Gradio).
2. Upload `hf_app/app.py` and `hf_app/requirements.txt`.
3. Upload the following files from your local `./results/phi2-grpo-final` to a folder named `model_adapters` in the Space:
   - `adapter_config.json`
   - `adapter_model.safetensors`
   - **Tokenizer Files** (Important since we modified padding):
     - `tokenizer_config.json`
     - `tokenizer.json`
     - `vocab.json`
     - `merges.txt`
     - `special_tokens_map.json`
     - `added_tokens.json`

