# med_rlhf/scripts/delete/delete_model.py

import os
import shutil
from med_rlhf.scripts.utils.model_utils import resolve_model_path


def delete_model():
    """ローカルのモデルを削除する関数"""
    try:
        user_input = input("削除するモデルのパスを指定してください: ").strip()

        # パスを解決
        model_path = resolve_model_path(user_input)
        if not model_path:
            print("[ERROR] モデルパスが指定されていません。")
            return

        if not os.path.exists(model_path):
            print(f"[ERROR] 指定されたパス '{model_path}' は存在しません。")
            return

        confirm = input(
            f"本当にモデル '{model_path}' を削除しますか？ (y/n): ").strip().lower()
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
