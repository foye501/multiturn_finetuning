from trl import setup_chat_format,SFTConfig, SFTTrainer,DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
from datasets import load_dataset, Dataset
import json
from accelerate import Accelerator
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
import torch

accelerator = Accelerator()

with open("cleaned_dialogues.json","r") as f:
    dataset = json.load(f)


with open("dialogues.jsonl", 'w', encoding='utf-8') as f:
    for conversation in dataset:
        f.write(json.dumps(conversation, ensure_ascii=False) + '\n')

datasets = load_dataset("json", data_files="dialogues.jsonl")

model = AutoModelForCausalLM.from_pretrained("my_model",
        torch_dtype=torch.float16, 
         load_in_4bit=True
        )
tokenizer = AutoTokenizer.from_pretrained("my_model")


def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize =False, add_generation_prompt =False) for convo in convos]
    return {"text": texts}

#model, tokenizer = setup_chat_format(model, tokenizer)
#tokenized_chat = tokenizer.apply_chat_template(dataset["train"][0]['messages'], tokenize=False, return_tensors="pt")


datasets = datasets["train"].map(formatting_prompts_func,batched =True)

instruction_template = "<|start_header_id|>user<|end_header_id|>"
response_template = "<|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)


# Ensure distinct tokens
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Resize model embedding size
    model.resize_token_embeddings(len(tokenizer),mean_resizing=False)

# Verify tokens
assert tokenizer.pad_token_id != tokenizer.eos_token_id, "pad_token_id and eos_token_id must be different!"

print("loading lora config")

lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM"  # Task type for causal language modeling
)



model = get_peft_model(model, lora_config)
model.gradient_checkpointing_disable()#  this one should be disable when load from lora.
model.config.use_cache = False
embedding_layer = model.get_input_embeddings()

# Check if embedding weights are trainable
print(embedding_layer.weight.requires_grad)  # Should return True if trainable
pad_token_id = tokenizer.pad_token_id
# Make embedding weights trainable explicitly if needed
embedding_layer.weight[pad_token_id].requires_grad = True  # Fine-tune padding embedding

model.print_trainable_parameters()
trainer = SFTTrainer(
    model,
    args=SFTConfig(output_dir="./test_lla", 
        per_device_train_batch_size =2,
        gradient_accumulation_steps=4,
        max_seq_length=2048,
        bf16=True,
        logging_steps=12,
        dataset_text_field = "text",
        optim = "adamw_hf" ),
    train_dataset=datasets,
    data_collator=collator,
)



trainer.train()





