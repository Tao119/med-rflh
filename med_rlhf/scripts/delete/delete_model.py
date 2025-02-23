# med_rlhf/scripts/delete/delete_model.py

import os
import shutil

def delete_model():
    """ローカルのモデルを削除する関数"""
    try:
        model_path = input("削除するモデルのパスを指定してください: ").strip()
        if not model_path:
            print("[ERROR] モデルパスが指定されていません。")
            return

        if not os.path.exists(model_path):
            print(f"[ERROR] 指定されたパス '{model_path}' は存在しません。")
            return

        confirm = input(f"本当にモデル '{model_path}' を削除しますか？ (y/n): ").strip().lower()
        if confirm != 'y':
            print("[INFO] モデルの削除をキャンセルしました。")
            return

        # ディレクトリまたはファイルを削除
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        else:
            os.remove(model_path)

        print(f"[INFO] モデル '{model_path}' を正常に削除しました。")
    except Exception as e:
        print(f"[ERROR] モデルの削除中にエラーが発生しました: {e}")
