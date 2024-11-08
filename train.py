# train.py
import os
import torch
import deepspeed
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.trainer import get_training_args, setup_trainer
from src.data import prepare_dataset
import logging

def setup_wandb(local_rank):
    """wandbの設定"""
    if local_rank <= 0:
        wandb.init(
            project="llama2-pretrain",
            entity="your-entity-name",  # wandbのユーザー名またはチーム名
            config={
                "model_name": "Llama-2-3b",
                "architecture": "Llama2",
                "dataset": "dataset_name",
            }
        )

def setup_distributed():
    """分散学習の初期設定"""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    return local_rank

def load_model_for_dataset(dataset_name, local_rank, logger):
    """各データセット用に新しいモデルをロード"""
    if local_rank <= 0:
        logger.info(f"Loading fresh model for {dataset_name}")
    
    return AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-3b",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=os.environ["HUGGING_FACE_HUB_TOKEN"]
    )

def main():
    # 分散学習のセットアップ
    local_rank = setup_distributed()
    logger = setup_logging(local_rank)

    # wandbのセットアップ
    setup_wandb(local_rank)
    
    # トークナイザーの取得（共通で使用）
    if local_rank <= 0:
        logger.info("Loading tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-3b",
        token=os.environ["HUGGING_FACE_HUB_TOKEN"]
    )
    
    # 学習するデータセット
    datasets_to_train = [
        {
            "name": "dataset1_name",
            "epochs": 1,
            "learning_rate": 1e-5,
        },
        {
            "name": "dataset2_name",
            "epochs": 1,
            "learning_rate": 1e-5,
        },
    ]
    
    # 各データセットで独立して学習
    for dataset_config in datasets_to_train:
        dataset_name = dataset_config["name"]
        if local_rank <= 0:
            logger.info(f"Starting independent training on {dataset_name}")

            if wandb.run is not None:
                wandb.finish()
            wandb.init(
                project="llama2-pretrain",
                name=f"train-{dataset_name}",
                config={
                    "model_name": "Llama-2-3b",
                    "dataset": dataset_name,
                    "epochs": dataset_config["epochs"],
                    "learning_rate": dataset_config["learning_rate"],
                }
            )
        
        # データセットごとに新しいモデルをロード
        model = load_model_for_dataset(dataset_name, local_rank, logger)
        
        # データセットの準備
        tokenized_dataset = prepare_dataset(dataset_name, tokenizer)
        
        # データセット固有の学習設定
        training_args = get_training_args(
            output_dir=f"./outputs/{dataset_name}",
            dataset_name=dataset_name,
            local_rank=local_rank,
            num_train_epochs=dataset_config["epochs"],
            learning_rate=dataset_config["learning_rate"]
        )
        
        # トレーナーの準備
        trainer = setup_trainer(model, training_args, tokenized_dataset, tokenizer)
        
        # 学習実行
        trainer.train()
        
        # モデルの保存（rank 0のプロセスのみ）
        if local_rank <= 0:
            logger.info(f"Saving model for {dataset_name}")
            trainer.save_model(f"./outputs/{dataset_name}/final")
        
        # メモリの解放
        del model
        del trainer
        torch.cuda.empty_cache()

        if local_rank <= 0:
            wandb.finish()

         # 最後にwandbをクリーンアップ
        if local_rank <= 0 and wandb.run is not None:
            wandb.finish()   

if __name__ == "__main__":
    main()