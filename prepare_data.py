from datasets import load_dataset
import os

def prepare_data():
    print("Loading OpenAssistant/oasst1 dataset...")
    # Load the train split
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    
    print(f"Original dataset size: {len(ds)}")
    
    # Filter for root messages (prompts)
    # in OASST1, root messages have parent_id == None
    prompts_ds = ds.filter(lambda x: x['parent_id'] is None)
    
    print(f"Number of prompts (root messages): {len(prompts_ds)}")
    
    # Rename 'text' to 'prompt' as expected by some trainers, or just keep it.
    # GRPO usually takes a list of "prompts". We'll verify this in the training script.
    # For now, let's just make sure we have the 'text' (prompt content) and 'lang'.
    
    prompts_ds = prompts_ds.select_columns(['text', 'lang', 'message_id'])
    prompts_ds = prompts_ds.rename_column('text', 'prompt')
    
    # Print language statistics
    lang_counts = {}
    for item in prompts_ds:
        l = item['lang']
        lang_counts[l] = lang_counts.get(l, 0) + 1
        
    print("\nPrompts by language:")
    for l, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{l}: {count}")
        
    # Save processed dataset
    output_path = "./data/oasst1_prompts"
    prompts_ds.save_to_disk(output_path)
    print(f"\nSaved processed dataset to {output_path}")

if __name__ == "__main__":
    prepare_data()
