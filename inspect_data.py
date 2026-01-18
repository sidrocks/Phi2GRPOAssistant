from datasets import load_dataset
import pandas as pd

# Load a small subset to inspect
try:
    ds = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
    print("Dataset loaded successfully.")
    
    print("First 5 examples:")
    i = 0
    for sample in ds:
        if i >= 5: break
        print(sample)
        print("-" * 20)
        i += 1
        
    # Check column names
    print(f"Features: {next(iter(ds)).keys()}")

except Exception as e:
    print(f"Error loading dataset: {e}")
