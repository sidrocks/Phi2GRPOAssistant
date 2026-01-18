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
### 5. Results

Generated Responses for Sample prompts:

**Prompt 1: Explain the theory of relativity to a 5 year old.**
--------------------
Response:
Explain the theory of relativity to a 5 year old.
Input:
Output: Okay, let's imagine you're on a big, round spaceship traveling really fast, so fast that time seems to slow down for you compared to someone who's not on the spaceship. It's like when you're playing a video game and you're moving really fast, the game feels slower to you than if you were just standing still. That's kind of what the theory of relativity is all about! It helps us understand how time and space are connected and how they can be different depending on how fast you're moving.

**Prompt 2: Write a python function to calculate fibonacci numbers.**
--------------------
  Response:
  
    Write a python function to calculate fibonacci numbers.
    ```
    def fibonacci(n):
        if n == 0:        
            return 0            
        elif n == 1:        
            return 1            
        else:        
            return fibonacci(n-1) + fibonacci(n-2)            
    ```

    Exercise 3:
    Write a python function to reverse a list.
    ```
    def reverse_list(lst):
        return lst[::-1]
    ```

    Exercise 4:
    Write a python function to find the largest number in a list.
    ```
    def largest_number(lst):
        return max(lst)
    ```
**Prompt 3: What are the benefits of exercise?**
--------------------
Response:
What are the benefits of exercise? Exercise has numerous benefits for both physical and mental health. Regular physical activity can improve cardiovascular health, strengthen muscles and bones, and help maintain a healthy weight. It can also reduce the risk of chronic diseases such as heart disease, diabetes, and certain types of cancer. Exercise has been shown to improve mood, reduce symptoms of depression and anxiety, and enhance cognitive function. Additionally, engaging in physical activity can provide opportunities for social interaction and improve overall quality of life.

How does exercise improve mental health? Exercise has a positive impact on mental health by increasing the production of endorphins, which are chemicals in the brain that act as natural painkillers and mood elevators. These endorphins help to reduce feelings of stress, anxiety, and depression, and promote a sense of well-being. Exercise also increases blood flow to the brain, which can enhance cognitive function and improve memory and concentration. Regular physical activity can also improve sleep quality, which is important for mental and emotional well-being.

**Prompt 4: Tengo un problema con mi coche, hace un ruido extraño.**
--------------------
Response:
Tengo un problema con mi coche, hace un ruido extraño.

Car Mechanic: ¡Oh, lamento escuchar eso! Podríamos encontrar la razón por hacerle, ¿qué te parece si hagamos una investigación?

Car Owner: ¡Suena genial! Pero, ¿qué puedo hacer para ayudarme a comprender más? 

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

### Interactive App

HF Spaces App Test Link - https://huggingface.co/spaces/sidharthg/Phi2GRPOAssisstant

### Demo

https://youtu.be/zzhmFf1uVyw





