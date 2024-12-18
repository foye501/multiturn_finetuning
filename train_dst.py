from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import json
from peft import LoraConfig, get_peft_model
import torch





dataset = load_dataset("json", data_files="processed_multi_turn.jsonl")


instruction_template = "<|start_header_id|>user<|end_header_id|>"
response_template = "<|start_header_id|>assistant<|end_header_id|>"



model = AutoModelForCausalLM.from_pretrained("my_model",torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("my_model")



# Ensure distinct tokens
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Resize model embedding size
    model.resize_token_embeddings(len(tokenizer),mean_resizing=False)

# Verify tokens
assert tokenizer.pad_token_id != tokenizer.eos_token_id, "pad_token_id and eos_token_id must be different!"

print("start  processing data ")
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

print("finished processing data ")



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
#model.gradient_checkpointing_disable()#  this one should be disable when load from lora.
#model.config.use_cache = False
embedding_layer = model.get_input_embeddings()

# Check if embedding weights are trainable
print(embedding_layer.weight.requires_grad)  # Should return True if trainable
pad_token_id = tokenizer.pad_token_id
# Make embedding weights trainable explicitly if needed
embedding_layer.weight[pad_token_id].requires_grad = True  # Fine-tune padding embedding

model.print_trainable_parameters()

print("finished config model")




trainer = SFTTrainer(
    model,
    args=SFTConfig(
         deepspeed="ds_config.json" ,
        output_dir="./sft_result"), 
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
