#!/usr/bin/env python3
# scripts/run_inference.py

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="対話形式")
    parser.add_argument("--query", type=str, default=None, help="1回の質問")
    args = parser.parse_args()

    # モデルを読み込む (models/base や micro-finetune後のモデル)
    # 省略。自作の load_model() 関数などで読み込むイメージ

    if args.interactive:
        print("[INFO] 対話形式で推論を行います。'quit' で終了。")
        while True:
            user_input = input("User> ")
            if user_input.lower() in ["quit", "exit"]:
                break
            # 推論処理して表示
            print(f"Bot> ... (model response)")
    elif args.query:
        # 1回限りの質問
        print(f"User> {args.query}")
        print("Bot> ... (model response)")
    else:
        print("[ERROR] 対話形式(--interactive)または--queryを指定してください。")


if __name__ == "__main__":
    main()
