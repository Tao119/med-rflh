#!/usr/bin/env python3
# main.py

import os
import sys
import subprocess
import shutil
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, HfApi, HfFolder
except ImportError:
    print("[WARNING] huggingface_hubがインストールされていません。")
    print("          `pip install huggingface_hub` を実行してください。")


def cmd_clone():
    """
    Hugging Faceモデルをローカルにクローン（ダウンロード）する処理
    """
    repo_id = input(
        "Hugging Face上のモデルRepo IDを入力してください (例: meta-llama/Llama-2-7b-hf): ").strip()
    local_dir = input("ダウンロード先ディレクトリ (例: models/base): ").strip()
    if not local_dir:
        local_dir = "models/base"

    os.makedirs(local_dir, exist_ok=True)
    print(f"[INFO] {repo_id} を {local_dir} にダウンロードします...")
    snapshot_download(repo_id=repo_id, local_dir=local_dir,
                      resume_download=True)
    print("[INFO] ダウンロード完了")


def cmd_upload():
    """
    ローカルディレクトリをHugging Faceにアップロードする処理
    """
    local_dir = input("アップロード対象のローカルモデルディレクトリ (例: models/base): ").strip()
    if not local_dir or not os.path.exists(local_dir):
        print("[ERROR] 指定したディレクトリが存在しません。")
        return

    repo_id = input(
        "アップロード先のHugging FaceのRepo ID (例: yourusername/mymodel): ").strip()
    if not repo_id:
        print("[ERROR] Repo IDが入力されていません。")
        return

    private_choice = input(
        "privateリポジトリにしますか？(y/n) [デフォルト=n]: ").strip().lower()
    private_flag = (private_choice == "y")

    # Hugging Face APIを使ってリポジトリを作成（存在しなければ）
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, private=private_flag)
    except Exception as e:
        print(f"[ERROR] リポジトリ作成に失敗しました: {e}")
        return

    # フォルダをアップロード
    commit_message = "Add model"
    print(
        f"[INFO] {local_dir} を {repo_id} にアップロードします (private={private_flag}).")
    try:
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        print("[INFO] アップロード完了。")
        print(f"URL: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[ERROR] アップロード失敗: {e}")


def cmd_train():
    """
    SFT or RLHFなど、学習を行うためのサンプル処理。
    実際には別のPythonスクリプトを呼び出したりTrainerを記述したり。
    """
    train_type = input("学習方式を選んでください: (1) SFT  (2) RLHF: ").strip()
    epochs = input("エポック数: ").strip()
    if not epochs.isdigit():
        epochs = "3"
    data_path = input("学習用データパス(例: data/sft_data.jsonl): ").strip()
    if not data_path:
        data_path = "data/sft_data.jsonl"

    if train_type == "1":
        print(f"[INFO] SFTを {epochs} エポックで実行. データ: {data_path}")
        # 実際の学習ロジック or scripts/train_sft.py を呼ぶなど
    else:
        print(f"[INFO] RLHFを {epochs} エポックで実行. データ: {data_path}")
        # 実際の学習ロジック or scripts/train_rlhf.py を呼ぶなど


def cmd_test():
    """
    モデル推論をテストするサンプル処理。
    実際にはrun_inference.pyなどをサブプロセスで呼ぶ場合が多い。
    """
    # 簡単に対話的に入力→モック応答する例
    while True:
        prompt = input("User> ").strip()
        if prompt.lower() in ["quit", "exit"]:
            break
        # 仮の応答
        print("Bot> ... (応答)")


def main_menu():
    """
    コマンド入力を受け取り、対応する処理に遷移。
    """
    while True:
        print("\n=== LLM CLI Menu ===")
        print("[clone] Hugging Faceからモデルをダウンロード")
        print("[upload] ローカルモデルをHugging Faceにアップロード")
        print("[train] 学習を開始 (SFT / RLHFなど)")
        print("[test] モデルのテスト (対話形式 など)")
        print("[exit] 終了")

        cmd = input("コマンドを入力: ").strip().lower()

        if cmd == "clone":
            cmd_clone()
        elif cmd == "upload":
            cmd_upload()
        elif cmd == "train":
            cmd_train()
        elif cmd == "test":
            cmd_test()
        elif cmd in ["exit", "quit"]:
            print("[INFO] 終了します。")
            break
        else:
            print("[ERROR] 無効なコマンドです。")


def main():
    main_menu()


if __name__ == "__main__":
    main()
