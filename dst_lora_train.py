import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import DataCollatorForCompletionOnlyLM

def main():
    # Initialize distributed training
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("my_model", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("my_model")
    def preprocess_function(example):
        return tokenizer(
            example["text"],
            max_length=1024,
            truncation=False,
            padding=False,
            )  
    model = model.to(local_rank)
    # Ensure special tokens are added
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer),mean_resizing=False)

    # Configure LoRA
    lora_config = LoraConfig(
        r=2,  # Rank
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Target layers for LoRA
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    # Freeze base model and train only LoRA parameters
    #for name, param in model.named_parameters():
    #   print(name)

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora" in name:  # LoRA parameters are tagged with "lora"
            param.requires_grad = True
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank])

    # Load dataset and define data collator
    dataset = load_dataset("json", data_files="processed_multi_turn.jsonl")
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template="<|start_header_id|>user<|end_header_id|>",
        response_template="<|start_header_id|>assistant<|end_header_id|>",
        tokenizer=tokenizer,
        mlm=False,
        padding=True,
        truncation=True
    )
    print(tokenized_dataset["train"][0])
    # Create DistributedSampler for the dataset
    train_sampler = DistributedSampler(
        tokenized_dataset["train"], num_replicas=dist.get_world_size(), rank=dist.get_rank()
    )
    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        sampler=train_sampler,
        batch_size=2,  # Adjust for per-device batch size
        collate_fn=collator,
    )
    


    # Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

    # Training loop
    model.train()
    for epoch in range(3):  # Adjust number of epochs
        train_sampler.set_epoch(epoch)  # Shuffle data for each epoch
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(local_rank) for k, v in batch.items()}  # Move batch to GPU
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            # Step optimizer and zero gradients
            optimizer.step()
            optimizer.zero_grad()

            # Print loss for monitoring
            if step % 10 == 0 and dist.get_rank() == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    # Save the model (only on rank 0)
    if dist.get_rank() == 0:
        model.module.save_pretrained("lora_finetuned_model")
        tokenizer.save_pretrained("lora_finetuned_model")

    # Clean up distributed process group
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
