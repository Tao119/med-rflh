#!/usr/bin/env python3
# scripts/train_sft.py

import os
import torch
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from utils.flash_attn_utils import apply_flash_attention, enable_gradient_checkpointing


def format_example(example):
    """
    'instruction','input','output' を1つのテキストにまとめる関数。
    必要に応じてプロンプトのフォーマットをカスタマイズしてください。
    """
    if example["input"]:
        return f"指示: {example['instruction']}\n追加情報: {example['input']}\n回答: {example['output']}"
    else:
        return f"指示: {example['instruction']}\n回答: {example['output']}"


def tokenize_function(examples, tokenizer):
    """バッチ単位で呼び出されるトークナイズ処理"""
    texts = [format_example(e) for e in examples["data"]]
    return tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length"
    )


def main():
    # === 1) パラメータ設定 ===
    base_model_name_or_path = "EleutherAI/gpt-neo-1.3B"  # 任意のCausalLM
    dataset_file = "data/processed/sft_dataset.jsonl"
    output_dir = "models/sft_qlora_final"
    merged_model_dir = "models/sft_merged_fp16"  # LoRA統合後の最終モデル出力先

    # === 2) データセットの読み込み & 前処理 ===
    #   sft_dataset.jsonl -> HF datasets でロードし、train/testに分割
    raw_data = load_dataset("json", data_files=dataset_file)["train"]
    dataset_split = raw_data.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    # datasetsの整形( {"instruction","input","output"} -> {"data":{...}} )してtokenize
    def wrap_dict(example):
        return {"data": {"instruction": example["instruction"],
                         "input": example["input"],
                         "output": example["output"]}}
    train_dataset = train_dataset.map(wrap_dict)
    eval_dataset = eval_dataset.map(wrap_dict)

    # === 3) モデル・トークナイザのロード (4bit量子化) ===
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading base model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        load_in_4bit=True,  # 4bit量子化を有効に
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb.QuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    )

    # === 4) Flash Attention & gradient checkpointing (任意で適用) ===
    model = apply_flash_attention(model)             # FlashAttention置換(サンプル)
    model = prepare_model_for_kbit_training(model)   # k-bit量子化に適した形へ
    model = enable_gradient_checkpointing(model)      # gradientチェックポイント有効化

    # === 5) LoRAパラメータの適用 ===
    # 注意: target_modulesはモデルによって異なる場合あり
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # === 6) トークナイズ & データセット整形 ===
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True, remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True, remove_columns=eval_dataset.column_names
    )

    train_dataset.set_format(type="torch")
    eval_dataset.set_format(type="torch")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # === 7) TrainingArguments ===
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=True,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        # DeepSpeed or FSDPの設定: 例: DeepSpeed
        deepspeed="config/ds_config.json",
        # fsdp="full_shard auto_wrap",    # もしaccelerate/FSDPを使う場合
        # fsdp_transformer_layer_cls_to_wrap="GPTNeoBlock",
        report_to="none"  # ロギングツールを使う場合 "wandb" など
    )

    # === 8) 学習開始 (Trainer) ===
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"[INFO] LoRA model saved to: {output_dir}")

    # === 9) ローカルテスト推論 (LoRA適用モデル) ===
    prompt = "高血圧の主な原因は何ですか？"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.9
        )
    print("\n=== Sample Inference (LoRA model) ===")
    print("PROMPT:", prompt)
    print("RESPONSE:", tokenizer.decode(outputs[0], skip_special_tokens=True))

    # === 10) LoRA差分をベースモデルにマージ (任意) ===
    # 4bit量子化された状態では正しくマージできない可能性があるため、
    # 一度fp16などでロードし直した上で merge_and_unload() を呼ぶのが一般的です。
    print("[INFO] Merging LoRA weights into base model for final fp16 model...")
    # 10-1) 元のベースモデルをfp16でロード
    base_model_fp16 = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # 10-2) LoRAモデルを読み込み
    lora_model = PeftModel.from_pretrained(
        base_model_fp16, output_dir, torch_dtype=torch.float16
    )
    # 10-3) 差分マージ
    merged_model = lora_model.merge_and_unload()

    # 10-4) 最終モデルをsave_pretrained (普通の全パラメータモデルとして保存)
    merged_model.save_pretrained(merged_model_dir)
    tokenizer.save_pretrained(merged_model_dir)
    print(f"[INFO] Merged full model saved to: {merged_model_dir}")

    # 10-5) 簡単にfp16モデルでテスト
    prompt2 = "インフルエンザと風邪の違いを教えてください。"
    inputs2 = tokenizer.encode(
        prompt2, return_tensors="pt").to(merged_model.device)
    with torch.no_grad():
        outputs2 = merged_model.generate(
            inputs2, max_new_tokens=128, do_sample=True, top_p=0.9, temperature=0.7
        )
    print("\n=== Sample Inference (Merged fp16 model) ===")
    print("PROMPT:", prompt2)
    print("RESPONSE:", tokenizer.decode(outputs2[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
