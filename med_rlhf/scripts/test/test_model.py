import threading
import itertools
import sys
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from med_rlhf.scripts.utils.model_utils import load_last_model_path, save_last_model_path, resolve_model_path
import torch


def spinner(stop_event):
    """アニメーション用スピナー関数"""
    for char in itertools.cycle(['|', '/', '-', '\\']):
        if stop_event.is_set():
            break
        sys.stdout.write(f'\rBot is thinking... {char}')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r' + ' ' * 30 + '\r')  # スピナーを消去


def test_model():
    """モデルのテスト（対話形式）"""
    try:
        last_model_path = load_last_model_path()
        user_input = input(
            f"テストするモデルのパスを指定してください (デフォルト: {last_model_path or '最後に使用したモデルがありません'}): ").strip()

        model_path = resolve_model_path(
            user_input) if user_input else last_model_path

        if not model_path:
            model_path = last_model_path
            if not model_path:
                print("[ERROR] モデルパスが指定されていません。")
                return

        print(f"[INFO] モデル {model_path} をロードしています...")
        save_last_model_path(model_path)  # 使用したモデルパスを保存

        # モデルとトークナイザーのロード
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        tokenizer.padding_side = "left"

        # Accelerateでロード
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(model_path)

        model = load_checkpoint_and_dispatch(
            model, model_path, device_map="auto", dtype=torch.float16 if device == "cuda" else torch.float32
        )

        print(f"[INFO] モデルロード完了。デバイス: {device}")
        print("[INFO] 'exit' と入力すると対話を終了します。\n")

        # 対話ループ
        while True:
            prompt = input("User> ").strip()
            if prompt.lower() in ["quit", "exit"]:
                print("[INFO] テストを終了します。")
                break

            # プロンプトをトークナイズ
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # スピナー開始
            stop_spinner = threading.Event()
            spinner_thread = threading.Thread(
                target=spinner, args=(stop_spinner,))
            spinner_thread.start()

            # モデルによる生成
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=200,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=0.7
                    )
            finally:
                # スピナーを停止
                stop_spinner.set()
                spinner_thread.join()

            # 応答をデコードして表示
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Bot> {response}")

    except Exception as e:
        print(f"[ERROR] テスト中にエラーが発生しました: {e}")
