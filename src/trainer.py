from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

def get_training_args(output_dir: str, dataset_name: str, local_rank: int):
    """学習設定の取得"""
    return TrainingArguments(
        output_dir=f"{output_dir}/{dataset_name}",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        deepspeed="config/ds_config.json",
        learning_rate=5e-5,
        warmup_steps=1000,
        weight_decay=0.01,
        local_rank=local_rank,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        report_to="wandb",
        logging_dir=f"./logs/{dataset_name}",  
        logging_first_step=True,  
    )

def setup_trainer(model, training_args, dataset, tokenizer):
    """トレーナーのセットアップ"""
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )