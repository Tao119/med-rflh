# scripts/upload/upload_model.py

from huggingface_hub import HfApi
import os

def upload_model():
    """ローカルディレクトリを Hugging Face にアップロード"""
    try:
        local_dir = input("アップロード対象のローカルモデルディレクトリ (例: models/sft/mymodel): ").strip()
        if not local_dir or not os.path.exists(local_dir):
            print("[ERROR] 指定したディレクトリが存在しません。")
            return

        repo_id = input("アップロード先のHugging FaceのRepo ID (例: yourusername/mymodel): ").strip()
        if not repo_id:
            print("[ERROR] Repo IDが入力されていません。")
            return

        private_choice = input("privateリポジトリにしますか？(y/n) [デフォルト=n]: ").strip().lower()
        private_flag = (private_choice == "y")

        api = HfApi()
        api.create_repo(repo_id=repo_id, exist_ok=True, private=private_flag)

        commit_message = "Add model"
        print(f"[INFO] {local_dir} を {repo_id} にアップロードします (private={private_flag}).")
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        print("[INFO] アップロード完了。")
        print(f"URL: https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"[ERROR] アップロード中にエラーが発生しました: {e}")
