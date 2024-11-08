from datasets import load_dataset

def prepare_dataset(dataset_name: str, tokenizer, max_length: int = 2048):
    """データセットの前処理"""
    dataset = load_dataset(dataset_name)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    return tokenized_dataset