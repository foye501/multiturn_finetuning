
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import json
from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

instruction_template = "<|start_header_id|>user<|end_header_id|>"
response_template = "<|start_header_id|>assistant<|end_header_id|>"


dataset = load_dataset("json", data_files="processed_multi_turn.jsonl")
print(dataset["train"][0])
print(dataset["train"][1])
print(dataset["train"][2])
print(dataset["train"][3])


instruction_template = "<|start_header_id|>user<|end_header_id|>"
response_template = "<|start_header_id|>assistant<|end_header_id|>"



#model = AutoModelForCausalLM.from_pretrained("my_model",torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("my_model")

    # Set a separate pad_token_id
tokenizer.pad_token_id = tokenizer.eos_token_id
def preprocess_function(examples):
    tokenized = tokenizer(
        examples["text"],  # Replace "text" with the correct column name
        truncation=True,
        padding=False,
        max_length=512
    )
    
    return tokenized

# Apply tokenization to the dataset
tokenized_dataset = dataset.map(preprocess_function, remove_columns=["text"], batched=False)


model = AutoModelForCausalLM.from_pretrained("my_model",torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("my_model")


model.config.use_cache = False

# Ensure distinct tokens
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Resize model embedding size
    model.resize_token_embeddings(len(tokenizer))

# Verify tokens
assert tokenizer.pad_token_id != tokenizer.eos_token_id, "pad_token_id and eos_token_id must be different!"


collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)


lora_config = LoraConfig(
    r=2,  # LoRA rank
    lora_alpha=16,
    target_modules= ["q_proj", "k_proj", "v_proj", "o_proj",
                    ],  # Target attention layers for LoRA
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"  # Task type for causal language modeling
)



model = get_peft_model(model, lora_config)

model.gradient_checkpointing_enable()
model.to("cuda:1")




# Initialize the data collator
data_collator = DataCollatorForCompletionOnlyLM(instruction_template =instruction_template,tokenizer=tokenizer,response_template =response_template,mlm=False )

samples = [tokenized_dataset["train"][i] for i in range(4)]

inputs = data_collator(samples)
inputs = {k: v.to("cuda:1") for k, v in inputs.items()}  # Move inputs to GPU if using CUDA

print(inputs)
outputs = model(**inputs)
loss = outputs.loss if "loss" in outputs else torch.tensor(0.0, requires_grad=True)

print(loss.requires_grad)  


for d in data_collator(samples)["labels"]:
    
    filtered_labels = [token_id for token_id in d if token_id != -100]

# Decode into text
    decoded_text = tokenizer.decode(filtered_labels, skip_special_tokens=True)

    print(decoded_text)


from torch.utils.data import DataLoader

dataloader = DataLoader(
    tokenized_dataset["train"],  # Use the tokenized dataset
    batch_size=4,  # Adjust batch size as needed
    collate_fn=data_collator
)

#print(dataloader)
# Inspect a single batch
#for batch in dataloader:
#    print(batch)
#    break

