
import os
from huggingface_hub import snapshot_download

# Hugging Face上のモデル名 (LLaMA 2, Falcon などの例)
# ここでは一例として "meta-llama/Llama-2-7b-hf" を挙げます
MODEL_REPO_ID = "meta-llama/Llama-2-7b-hf"

# ダウンロード先ディレクトリ
LOCAL_MODEL_DIR = "models/base"


def main():
    # 既にダウンロード済みかチェック
    if os.path.exists(LOCAL_MODEL_DIR) and len(os.listdir(LOCAL_MODEL_DIR)) > 0:
        print(f"[INFO] {LOCAL_MODEL_DIR} には既にモデルが存在します。再ダウンロードは行いません。")
    else:
        print(
            f"[INFO] Downloading model from {MODEL_REPO_ID} to {LOCAL_MODEL_DIR}...")
        snapshot_download(
            repo_id=MODEL_REPO_ID,
            local_dir=LOCAL_MODEL_DIR,
            # ↓ 研究ライセンスモデル(LLaMAなど)の場合、アクセストークンが必要になる場合も
            # token="<YOUR_HF_ACCESS_TOKEN>",
            # 以下オプション：存在しているファイルを再ダウンロードしない設定
            resume_download=True,
        )
        print("[INFO] Download complete.")


if __name__ == "__main__":
    main()
