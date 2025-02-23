# scripts/clone/clone_model.py

from huggingface_hub import snapshot_download, HfFolder, login
import os
from med_rlhf.scripts.utils.model_utils import save_last_model_path

def clone_model():
    """Hugging Face モデルをローカルにクローンする"""
    try:
        # トークンの確認
        token = HfFolder.get_token()
        if not token:
            print("[INFO] Hugging Face にログインする必要があります。")
            token = input("Hugging Face アクセストークンを入力してください: ").strip()
            if not token:
                print("[ERROR] トークンが入力されていません。クローンを中止します。")
                return

            login(token=token)
            print("[INFO] Hugging Face にログインしました。")

        # モデルリポジトリID取得
        repo_id = input("Hugging Face上のモデルRepo IDを入力してください (例: meta-llama/Llama-2-7b-hf): ").strip()
        if not repo_id:
            print("[ERROR] Repo IDが入力されていません。")
            return

        # クローン先パス設定
        repo_dir_name = repo_id.replace("/", "-")
        local_dir = f"models/base/{repo_dir_name}"
        os.makedirs(local_dir, exist_ok=True)

        print(f"[INFO] {repo_id} を {local_dir} にダウンロードします...")

        # モデルのクローン
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            resume_download=True,
            use_auth_token=token
        )

        print("[INFO] ダウンロード完了")
        save_last_model_path(local_dir)  # 最後に使用したモデルパスを保存

    except Exception as e:
        print(f"[ERROR] モデルのクローン中にエラーが発生しました: {e}")
