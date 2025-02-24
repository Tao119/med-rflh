from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from med_rlhf.scripts.utils.model_utils import resolve_model_path, save_last_model_path, load_last_model_path
import torch
import os
import wandb


def train_rlhf():
    """Reinforcement Learning from Human Feedback (RLHF) の実行"""
    try:
        # モデルの選択
        last_model_path = load_last_model_path()
        user_model_input = input(
            f"ベースモデル名またはパスを入力してください (デフォルト: {last_model_path or '最後に使用したモデルがありません'}): ").strip()
        model_path = resolve_model_path(
            user_model_input) if user_model_input else last_model_path
        save_last_model_path(model_path)

        # 学習データの選択
        data_input = input(
            "学習データのパスを入力してください (デフォルト: rlhf_data.jsonl): ").strip()
        data_path = f"data/{data_input}" if not data_input.startswith(
            "data/") else data_input

        # 出力ディレクトリ
        output_name = input(
            "保存するモデルの名前 (デフォルト: rlhf_model): ").strip() or "rlhf_model"
        output_dir = f"models/trained/{output_name}"
        os.makedirs(output_dir, exist_ok=True)

        # 学習パラメータ
        epochs = int(input("エポック数 (デフォルト: 3): ").strip() or 3)
        batch_size = int(input("バッチサイズ (デフォルト: 4): ").strip() or 4)
        learning_rate = float(input("学習率 (デフォルト: 5e-5): ").strip() or 5e-5)

        print(f"[INFO] モデル {model_path} をロードしています...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)

        # データセットのロード
        if not os.path.exists(data_path):
            print(f"[ERROR] 指定したデータセットが見つかりません: {data_path}")
            return

        print(f"[INFO] データセット {data_path} をロードしています...")
        dataset = load_dataset("json", data_files=data_path, split="train")

        # `wandb` ログイン
        wandb.init(project="RLHF_Training", name=output_name)

        # PPO 設定
        config = PPOConfig(
            model_name=model_path,
            learning_rate=learning_rate,
            batch_size=batch_size,
            ppo_epochs=epochs,
            log_with="wandb"
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
                query_tensors = tokenizer(
                    batch["text"], return_tensors="pt", truncation=True, padding=True).input_ids
                response_tensors = model.generate(
                    query_tensors, max_length=128)
                rewards = torch.tensor(
                    [1.0] * response_tensors.shape[0])  # ダミー報酬
                ppo_trainer.step(query_tensors, response_tensors, rewards)

        # モデル保存
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[INFO] RLHF 学習完了。モデルは {output_dir} に保存されました。")

    except Exception as e:
        print(f"[ERROR] RLHF 学習中にエラーが発生しました: {e}")
