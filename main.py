#!/usr/bin/env python3
"""
main.py

本スクリプトを実行すると、以下の手順を対話形式で行います:
1) venv (仮想環境) の作成やアクティベーションの確認
2) requirements.txt のインストール
3) モデル選択 (Hugging Faceからダウンロード or ローカルパス指定)
4) 学習種類 (SFT or 強化学習) の選択、パラメータ設定
5) 学習スクリプトの実行
6) テスト (推論) の実行
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_and_setup_venv(venv_dir="venv"):
    """
    1) 指定したvenvディレクトリが存在するかチェック。
    2) 存在しなければ作成。
    3) 仮想環境がアクティブかどうかを簡易チェック。
       もしアクティブでなければ、インストール前に案内メッセージを表示。
    """
    venv_path = Path(venv_dir)
    if not venv_path.exists():
        print("[INFO] venvが存在しません。仮想環境を作成します...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)])

    # 仮想環境がアクティブかどうか簡易判定
    # (本当はもっと厳密にチェックする必要があるかもしれません)
    if os.environ.get("VIRTUAL_ENV") is None:
        print("[WARNING] 仮想環境がアクティブではないようです。")
        print("  1) source venv/bin/activate (Mac/Linuxの場合)")
        print("  2) .\\venv\\Scripts\\activate (Windowsの場合)")
        print("上記を実行してから再度このスクリプトを実行することを推奨します。")


def install_requirements(requirements_file="requirements.txt"):
    """
    requirements.txt を pip installする。
    """
    if not os.path.exists(requirements_file):
        print(f"[WARNING] {requirements_file} が見つかりません。スキップします。")
        return
    print(f"[INFO] pip install -r {requirements_file} を実行します...")
    subprocess.run([sys.executable, "-m", "pip",
                   "install", "-r", requirements_file])


def clone_or_select_model():
    """
    ユーザーに「Hugging Faceからのダウンロード」か「ローカルディレクトリを使用」か選択させる。
    """
    print("モデルの選択方法を選んでください:")
    print("1) Hugging Face からダウンロード (例: meta-llama/Llama-2-7b-hf)")
    print("2) ローカルのモデルディレクトリを指定")
    choice = input("選択(1 or 2): ").strip()

    model_dir = "models/base"
    if choice == "1":
        repo_id = input(
            "Hugging Face上のrepo_idを入力してください (例: meta-llama/Llama-2-7b-hf): ").strip()
        # huggingface_hub の snapshot_downloadを使用
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("[ERROR] huggingface_hub がインストールされていません。")
            print("       pip install huggingface_hub でインストールしてください。")
            sys.exit(1)

        os.makedirs(model_dir, exist_ok=True)
        print(f"[INFO] {repo_id} を {model_dir} にダウンロードします...")
        snapshot_download(repo_id=repo_id, local_dir=model_dir,
                          resume_download=True)
        print("[INFO] ダウンロード完了。")

    elif choice == "2":
        local_path = input("ローカルのモデルディレクトリパスを入力してください: ").strip()
        if not os.path.exists(local_path):
            print("[ERROR] 指定パスが存在しません。")
            sys.exit(1)
        # models/base ディレクトリにコピー or シンボリックリンクなど
        if os.path.abspath(local_path) != os.path.abspath(model_dir):
            os.makedirs("models", exist_ok=True)
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            shutil.copytree(local_path, model_dir)
        print(f"[INFO] ローカルモデル {local_path} を {model_dir} として利用可能にしました。")

    else:
        print("[ERROR] 無効な選択です。終了します。")
        sys.exit(1)

    return model_dir


def choose_training():
    """
    ユーザに対し、SFT or RLHF どちらを行うか選択させる。
    あわせてエポック数や学習データディレクトリの入力を受け取る。
    """
    print("学習タイプを選んでください:")
    print("1) SFT (教師あり微調整)")
    print("2) 強化学習 (RLHFなど)")
    choice = input("選択(1 or 2): ").strip()

    epochs = input("学習エポック数を入力してください (デフォルト:3): ").strip()
    if not epochs.isdigit():
        epochs = "3"  # デフォルト

    data_dir = input(
        "学習データのディレクトリ or ファイルパスを入力してください (例: data/sft_data.jsonl): ").strip()
    if not data_dir:
        data_dir = "data/sft_data.jsonl"

    return choice, int(epochs), data_dir


def run_training(train_type, epochs, data_file):
    """
    SFT or RLHFの学習スクリプトをサブプロセスで呼ぶ。
    ここでは scripts/train_sft.py や scripts/train_rlhf.py を想定。
    """
    if train_type == "1":
        print(f"[INFO] SFTを {epochs} エポックで実行します。データ: {data_file}")
        cmd = [
            sys.executable, "scripts/train_sft.py",
            "--epochs", str(epochs),
            "--data", data_file
        ]
        subprocess.run(cmd)
    else:
        print(f"[INFO] 強化学習(RLHF)を {epochs} エポック想定で実行します。データ: {data_file}")
        cmd = [
            sys.executable, "scripts/train_rlhf.py",
            "--epochs", str(epochs),
            "--data", data_file
        ]
        subprocess.run(cmd)


def test_inference():
    """
    学習済みモデルを対話形式 or コンテクスト付き推論を行う。
    ここでは scripts/run_inference.py を呼び、簡易な対話を行うことを想定。
    """
    print("テスト形式を選んでください:")
    print("1) 対話形式 (対話を繰り返す)")
    print("2) 1回のコンテクスト入力")
    choice = input("選択(1 or 2): ").strip()

    if choice == "1":
        subprocess.run(
            [sys.executable, "scripts/run_inference.py", "--interactive"])
    else:
        context = input("コンテクスト or 質問を入力してください: ")
        subprocess.run(
            [sys.executable, "scripts/run_inference.py", "--query", context])


def main():
    print("=== LLM Training & Testing CLI ===")
    # 1) venvチェック & requirementsインストール
    check_and_setup_venv("venv")
    install_requirements("requirements.txt")

    # 2) モデル選択
    model_dir = clone_or_select_model()
    print(f"[INFO] モデルディレクトリ: {model_dir}")

    # 3) 学習 or スキップを選択
    do_train = input("学習を行いますか？ (y/n): ").strip().lower()
    if do_train == "y":
        train_type, epochs, data_file = choose_training()
        run_training(train_type, epochs, data_file)

    # 4) テスト実行
    do_test = input("テスト推論を行いますか？ (y/n): ").strip().lower()
    if do_test == "y":
        test_inference()

    print("[INFO] アプリ終了。お疲れ様でした。")


if __name__ == "__main__":
    main()
