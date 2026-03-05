"""
train.py — Fine-tune Qwen3-4B-Instruct on function calling data using LoRA (QLoRA).

This script:
1. Loads the pre-processed dataset (from prepare_dataset.py)
2. Loads Qwen3-4B-Instruct in 4-bit quantization
3. Applies LoRA adapters for parameter-efficient fine-tuning
4. Trains using Hugging Face's SFTTrainer
5. Saves the final merged model (base + adapter) for inference

Usage:
    python train.py

Prerequisites:
    - Run `python prepare_dataset.py` first to generate the processed dataset
    - GPU with >= 8GB VRAM recommended (QLoRA 4-bit)
"""

import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from config import ModelConfig, DataConfig, LoRAConfig, TrainingConfig


def main():
    model_cfg = ModelConfig()
    data_cfg = DataConfig()
    lora_cfg = LoRAConfig()
    train_cfg = TrainingConfig()

    # ----------------------------------------------------------------
    # 1. Load processed dataset
    # ----------------------------------------------------------------
    data_path = os.path.join(data_cfg.processed_data_dir, "processed_dataset")
    print(f"Loading processed dataset from: {data_path}")
    dataset = load_from_disk(data_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]
    print(f"  Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # ----------------------------------------------------------------
    # 2. Load tokenizer
    # ----------------------------------------------------------------
    print(f"\nLoading tokenizer: {model_cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----------------------------------------------------------------
    # 3. Load model with 4-bit quantization (QLoRA)
    # ----------------------------------------------------------------
    print(f"Loading model: {model_cfg.model_name}")

    if model_cfg.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("  Using 4-bit quantization (QLoRA)")
    else:
        bnb_config = None
        print("  Loading in full precision")

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Enable gradient checkpointing to save memory
    if train_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Prepare model for k-bit training (freeze base, enable LoRA-compatible layers)
    if model_cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # ----------------------------------------------------------------
    # 4. Apply LoRA adapters
    # ----------------------------------------------------------------
    print(f"  Applying LoRA (r={lora_cfg.r}, alpha={lora_cfg.lora_alpha})")
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        task_type=lora_cfg.task_type,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ----------------------------------------------------------------
    # 5. Set up training arguments
    # ----------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_ratio=train_cfg.warmup_ratio,
        weight_decay=train_cfg.weight_decay,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        eval_strategy="steps",
        eval_steps=train_cfg.eval_steps,
        save_total_limit=train_cfg.save_total_limit,
        fp16=train_cfg.fp16,
        bf16=train_cfg.bf16,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        optim=train_cfg.optim,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        report_to=train_cfg.report_to,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    # ----------------------------------------------------------------
    # 6. Data collator
    # ----------------------------------------------------------------
    # Use a standard data collator that pads to the longest in the batch
    # and sets padding token labels to -100 (ignored in loss)
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer,
    )

    # ----------------------------------------------------------------
    # 7. Initialize trainer
    # ----------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # ----------------------------------------------------------------
    # 8. Train!
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    # ----------------------------------------------------------------
    # 9. Save the LoRA adapter
    # ----------------------------------------------------------------
    adapter_path = os.path.join(train_cfg.output_dir, "lora_adapter")
    print(f"\nSaving LoRA adapter to: {adapter_path}")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    # ----------------------------------------------------------------
    # 10. Merge adapter with base model and save final model
    # ----------------------------------------------------------------
    print(f"Merging adapter with base model...")
    merged_model = model.merge_and_unload()

    final_path = os.path.join(train_cfg.output_dir, "final_model")
    print(f"Saving merged model to: {final_path}")
    merged_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print("\nTraining complete!")
    print(f"  LoRA adapter: {adapter_path}")
    print(f"  Merged model: {final_path}")
    print(f"\nNext step: python inference.py")

    # ----------------------------------------------------------------
    # Optional: Push to Hugging Face Hub
    # ----------------------------------------------------------------
    # Uncomment the lines below to push your model to HF Hub:
    #
    # HF_REPO_ID = "<your-username>/qwen3-4b-function-calling"
    # print(f"\nPushing to HF Hub: {HF_REPO_ID}")
    # merged_model.push_to_hub(HF_REPO_ID)
    # tokenizer.push_to_hub(HF_REPO_ID)
    # print("Pushed successfully!")


if __name__ == "__main__":
    main()
