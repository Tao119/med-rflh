# med_rlhf/main.py

from med_rlhf.scripts.train.train_sft import train_sft
from med_rlhf.scripts.train.train_rlhf import train_rlhf
from med_rlhf.scripts.clone.clone_model import clone_model
from med_rlhf.scripts.upload.upload_model import upload_model
from med_rlhf.scripts.test.test_model import test_model
from med_rlhf.scripts.delete.delete_model import delete_model  # 追加


def main_menu():
    """CLI メニューを表示してコマンドを実行"""
    while True:
        try:
            print("\n=== LLM CLI Menu ===")
            print("[clone] Hugging Faceからモデルをダウンロード")
            print("[upload] ローカルモデルをHugging Faceにアップロード")
            print("[sft] SFT 学習を開始")
            print("[rlhf] RLHF 学習を開始")
            print("[test] モデルのテスト (対話形式 など)")
            print("[delete] モデルを削除")  # 追加
            print("[exit] 終了")

            cmd = input("コマンドを入力: ").strip().lower()

            if cmd == "clone":
                clone_model()
            elif cmd == "upload":
                upload_model()
            elif cmd == "sft":
                train_sft()
            elif cmd == "rlhf":
                train_rlhf()
            elif cmd == "test":
                test_model()
            elif cmd == "delete":  # 追加
                delete_model()
            elif cmd in ["exit", "quit"]:
                print("[INFO] 終了します。")
                break
            else:
                print("[ERROR] 無効なコマンドです。")
        except Exception as e:
            print(f"[ERROR] メニュー操作中に予期しないエラーが発生しました: {e}")


def main():
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n[INFO] プログラムを終了します。")
    except Exception as e:
        print(f"[ERROR] 予期しないエラー: {e}")


if __name__ == "__main__":
    main()
