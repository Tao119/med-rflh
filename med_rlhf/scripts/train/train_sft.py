# med_rlhf/scripts/train/train_sft.py

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import os

def train_sft():
    """Supervised Fine-Tuning (SFT) の実行"""
    try:
        # パラメータ入力
        model_name = input("ベースモデル名またはパスを入力してください (例: gpt2): ").strip()
        data_path = input("学習データのパス (例: data/sft_data.jsonl): ").strip()
        output_dir = "models/trained/sft_model"
        os.makedirs(output_dir, exist_ok=True)

        epochs = int(input("エポック数 (デフォルト: 3): ").strip() or 3)
        batch_size = int(input("バッチサイズ (デフォルト: 4): ").strip() or 4)
        learning_rate = float(input("学習率 (デフォルト: 5e-5): ").strip() or 5e-5)

        print(f"[INFO] モデル {model_name} をロードしています...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # データセットのロード
        print(f"[INFO] データセット {data_path} をロードしています...")
        dataset = load_dataset("json", data_files=data_path, split="train")

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # データコラレーター
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # トレーニング引数
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_dir='./logs',
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2
        )

        # トレーナーの作成
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            data_collator=data_collator
        )

        # 学習実行
        print("[INFO] SFT 学習を開始します...")
        trainer.train()

        # モデル保存
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[INFO] SFT 学習完了。モデルは {output_dir} に保存されました。")

    except Exception as e:
        print(f"[ERROR] SFT 学習中にエラーが発生しました: {e}")
