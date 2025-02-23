# med_rlhf/scripts/train/train_rlhf.py

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
import torch
import os

def train_rlhf():
    """Reinforcement Learning from Human Feedback (RLHF) の実行"""
    try:
        # パラメータ入力
        model_name = input("ベースモデル名またはパスを入力してください (例: gpt2): ").strip()
        data_path = input("学習データのパス (例: data/rlhf_data.jsonl): ").strip()
        output_dir = "models/trained/rlhf_model"
        os.makedirs(output_dir, exist_ok=True)

        epochs = int(input("エポック数 (デフォルト: 3): ").strip() or 3)
        batch_size = int(input("バッチサイズ (デフォルト: 4): ").strip() or 4)
        learning_rate = float(input("学習率 (デフォルト: 5e-5): ").strip() or 5e-5)

        print(f"[INFO] モデル {model_name} をロードしています...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

        # データセットのロード
        print(f"[INFO] データセット {data_path} をロードしています...")
        dataset = load_dataset("json", data_files=data_path, split="train")

        # PPO 設定
        config = PPOConfig(
            model_name=model_name,
            learning_rate=learning_rate,
            batch_size=batch_size,
            ppo_epochs=epochs,
            log_with=None
        )

        # トレーナーの初期化
        ppo_trainer = PPOTrainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset
        )

        # 学習ループ
        print("[INFO] RLHF 学習を開始します...")
        for epoch in range(epochs):
            print(f"[INFO] エポック {epoch + 1}/{epochs}")
            for batch in dataset:
                query_tensors = tokenizer(batch["text"], return_tensors="pt", truncation=True, padding=True).input_ids
                response_tensors = model.generate(query_tensors, max_length=128)
                rewards = torch.tensor([1.0] * response_tensors.shape[0])  # ダミー報酬

                # PPO ステップ
                ppo_trainer.step(query_tensors, response_tensors, rewards)

        # モデル保存
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[INFO] RLHF 学習完了。モデルは {output_dir} に保存されました。")

    except Exception as e:
        print(f"[ERROR] RLHF 学習中にエラーが発生しました: {e}")
