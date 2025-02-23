from huggingface_hub import HfApi
from med_rlhf.scripts.utils.model_utils import resolve_model_path
import os


def upload_model():
    """ローカルディレクトリを Hugging Face にアップロード"""
    try:
        # モデルパス入力（models/ 省略可）
        user_input_path = input(
            "アップロード対象のローカルモデルディレクトリ (例: trained/my-model): ").strip()
        local_dir = resolve_model_path(user_input_path)

        if not local_dir or not os.path.exists(local_dir):
            print(f"[ERROR] 指定したディレクトリが存在しません: {local_dir}")
            return

        # Hugging Face Repo ID 入力
        repo_id = input(
            "アップロード先の Hugging Face の Repo ID (例: yourusername/mymodel): ").strip()
        if not repo_id:
            print("[ERROR] Repo ID が入力されていません。")
            return

        # private 設定確認
        private_choice = input(
            "private リポジトリにしますか？(y/n) [デフォルト=n]: ").strip().lower()
        private_flag = (private_choice == "y")

        # Hugging Face API インスタンス生成
        api = HfApi()

        # リポジトリ作成（存在しない場合）
        print(f"[INFO] Hugging Face リポジトリ '{repo_id}' を作成・確認中...")
        api.create_repo(repo_id=repo_id, exist_ok=True, private=private_flag)

        # アップロード処理
        commit_message = "Add model"
        print(
            f"[INFO] {local_dir} を {repo_id} にアップロードします (private={private_flag})...")
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            commit_message=commit_message,
        )

        # 完了メッセージ
        print("[INFO] アップロード完了！")
        print(f"[INFO] モデル URL: https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"[ERROR] アップロード中にエラーが発生しました: {e}")
