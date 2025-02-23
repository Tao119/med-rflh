#!/usr/bin/env python3
# scripts/train_rlhf.py

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

# 独自ユーティリティ
from utils.flash_attn_utils import apply_flash_attention, enable_gradient_checkpointing


def main():
    # --- 1) 事前にSFT済みのモデル(LoRA込み)を読み込む ---
    sft_model_path = "models/sft_qlora_flashattn"

    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # RLHF用としてValueHeadを備えたモデルへ
    base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        sft_model_path, torch_dtype=torch.float16)

    # (任意) FlashAttentionやGradientCheckointing
    base_model = apply_flash_attention(base_model)
    enable_gradient_checkpointing(base_model)

    base_model.eval()

    # --- 2) 報酬モデルの準備(仮) ---
    #  本来はペアワイズ比較データ等から学習した報酬モデルをロードする
    #  ここでは簡易に同じSFTモデルを "疑似報酬モデル" として流用。実務では別学習が必要。
    reward_model = AutoModelForCausalLM.from_pretrained(
        sft_model_path, torch_dtype=torch.float16)
    reward_model.eval()

    # --- 3) PPOの設定 ---
    ppo_config = PPOConfig(
        batch_size=2,
        gradient_accumulation_steps=4,
        model_name=sft_model_path,
        learning_rate=1e-5,
        log_with=None,  # or "wandb"
        mini_batch_size=1
    )

    # PPOTrainerの初期化
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=base_model,
        tokenizer=tokenizer,
        ref_model=base_model,  # 参考モデル
        reward_model=reward_model  # 報酬モデル
    )

    # --- 4) データ準備(例: RLHF用データセット) ---
    #   例えばユーザーqueryのリストを用意。実際には大量のプロンプトが必要。
    #   RLHFの場合は比較データ or preferenceモデリングなどで報酬を計算するフローが一般的。
    prompts = [
        "インフルエンザと普通の風邪の違いを教えてください。",
        "高血圧の主な原因は何ですか？"
    ]

    # --- 5) PPO学習ループ(簡易例) ---
    for epoch in range(1):
        for prompt in prompts:
            query_tensor = tokenizer.encode(
                prompt, return_tensors="pt").to(base_model.device)
            # モデルによる応答生成
            response_tensor = ppo_trainer.generate(
                query_tensor,
                max_new_tokens=64,
                top_k=50,
                top_p=0.9,
                temperature=1.0
            )
            # 応答をテキスト化
            response_text = tokenizer.decode(
                response_tensor[0], skip_special_tokens=True)

            # 報酬計算(ここではreward_modelでスコア推定する例) -> TRLが内部で処理
            # もしくは custom コールバックで好ましさを計算し、 reward = f(response_text) で PPOTrainer.step
            reward = [1.0]  # ダミー報酬

            # PPOアップデート
            stats = ppo_trainer.step(
                query_tensor[0], response_tensor[0], reward)

            print("PROMPT:", prompt)
            print("RESPONSE:", response_text)
            print("REWARD:", reward, "PPO stats:", stats)

    # PPO後のモデルを保存(ValueHead付きの重みなど)
    ppo_trainer.model.save_pretrained("models/ppo_rlhf_output")
    tokenizer.save_pretrained("models/ppo_rlhf_output")


if __name__ == "__main__":
    main()
