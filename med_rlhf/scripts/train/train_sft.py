from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from med_rlhf.scripts.utils.model_utils import resolve_model_path, save_last_model_path, load_last_model_path
import os


def train_sft():
    """Supervised Fine-Tuning (SFT) の実行"""
    try:
        # パラメータ入力
        last_model_path = load_last_model_path()
        user_model_input = input(
            f"ベースモデル名またはパスを入力してください (デフォルト: {last_model_path or '最後に使用したモデルがありません'}): ").strip()
        model_path = resolve_model_path(
            user_model_input) if user_model_input else last_model_path
        save_last_model_path(model_path)

        data_path = input(
            "学習データのパス (デフォルト: data/sft_data.jsonl): ").strip() or "data/sft_data.jsonl"
        output_dir = "models/trained/sft_model"
        os.makedirs(output_dir, exist_ok=True)

        epochs = int(input("エポック数 (デフォルト: 3): ").strip() or 3)
        batch_size = int(input("バッチサイズ (デフォルト: 4): ").strip() or 4)
        learning_rate = float(input("学習率 (デフォルト: 5e-5): ").strip() or 5e-5)

        print(f"[INFO] モデル {model_path} をロードしています...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        # データセットのロード
        if not os.path.exists(data_path):
            print(f"[ERROR] 指定したデータセットが見つかりません: {data_path}")
            return

        print(f"[INFO] データセット {data_path} をロードしています...")
        dataset = load_dataset("json", data_files=data_path, split="train")

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # データコラレーター
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False)

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
