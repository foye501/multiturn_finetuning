from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load the base model
base_model_name = "Undi95/Llama-3-LewdPlay-8B-evo"  # Replace with your base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)



tokenizer = AutoTokenizer.from_pretrained(base_model_name)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Resize the model's embeddings to match the tokenizer size
base_model.resize_token_embeddings(len(tokenizer))

# Verify tokens
assert tokenizer.pad_token_id != tokenizer.eos_token_id, "pad_token_id and eos_token_id must be different!"



# Load the LoRA-adapted model
lora_model_name = "test_lla/checkpoint-400"
lora_model = PeftModel.from_pretrained(base_model, lora_model_name)

# Merge the LoRA weights into the base model
merged_model = lora_model.merge_and_unload()

# Save the merged model as a new base model
save_directory = "./emoji_combine0228"
merged_model.save_pretrained(save_directory)

# Save the tokenizer if needed
tokenizer.save_pretrained(save_directory)

print(f"Merged model saved to {save_directory}")
