from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import json
from peft import LoraConfig, get_peft_model
import torch


dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

print(dataset[0])


dataset = load_dataset("json", data_files="processed_multi_turn.jsonl")


instruction_template = "<|start_header_id|>user<|end_header_id|>"
response_template = "<|start_header_id|>assistant<|end_header_id|>"



model = AutoModelForCausalLM.from_pretrained("my_model",torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("my_model")


model.to("cuda:0")

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
    args=SFTConfig(output_dir="/tmp", 
        per_device_train_batch_size =1,
        max_seq_length=2048,
        fp16=True,
        dataset_text_field = "text",
        optim = "adamw_8bit" ),
    train_dataset=dataset["train"],
    data_collator=collator,
)





batch = next(iter(trainer.get_train_dataloader()))
batch = {k: v.to("cuda:0") for k, v in batch.items()}  # Ensure batch is on correct device

outputs = model(**batch)
loss = outputs.loss
logits = outputs.logits
print("Logits requires grad:", logits.requires_grad)

print("Loss:", loss)
print("Loss requires grad:", loss.requires_grad)



trainer.train()
