#!/usr/bin/env bash

set -e  # エラー発生時に即終了

# ================================
# 設定
# ================================
REQUIRED_PYTHON_VERSION="3.12"  # 使用する Python バージョン
VENV_DIR=".venv"                # 仮想環境ディレクトリ
DEFAULT_MODEL_REPO="rinna/deepseek-r1-distill-qwen2.5-bakeneko-32b"
DEFAULT_MODEL_DIR="models/base/rinna-deepseek-r1-distill-qwen2.5-bakeneko-32b"

# ================================
# 1. Python バージョンの確認
# ================================
echo "[INFO] Python バージョンを確認しています..."

if ! command -v python$REQUIRED_PYTHON_VERSION &> /dev/null; then
    echo "[ERROR] Python $REQUIRED_PYTHON_VERSION がインストールされていません。"
    echo "Ubuntu 24.04 では次のコマンドでインストールできます:"
    echo "  sudo apt update && sudo apt install python$REQUIRED_PYTHON_VERSION python$REQUIRED_PYTHON_VERSION-venv python$REQUIRED_PYTHON_VERSION-dev"
    exit 1
fi

# Python バージョンの確認
CURRENT_PYTHON_VERSION=$(python$REQUIRED_PYTHON_VERSION --version 2>&1 | awk '{print $2}')
echo "[INFO] 使用する Python バージョン: $CURRENT_PYTHON_VERSION"

# ================================
# 2. 仮想環境の作成
# ================================
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] 仮想環境を作成します: $VENV_DIR"
    python$REQUIRED_PYTHON_VERSION -m venv "$VENV_DIR"
else
    echo "[INFO] 仮想環境は既に存在します: $VENV_DIR"
fi

# 仮想環境のアクティベート
source "$VENV_DIR/bin/activate"

# ================================
# 3. 必要なパッケージのインストール
# ================================
echo "[INFO] pip, setuptools, wheel をアップグレードします..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1 && echo "[INFO] パッケージが最新です。"

# ================================
# 4. GPU/CPU 環境の判定と依存関係のインストール
# ================================
echo "[INFO] GPU 環境の有無を確認しています..."
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "[INFO] GPU 環境が検出されました。GPU バージョンをインストールします。"
    pip install .[gpu] --quiet && echo "[INFO] GPU 依存関係のインストールが完了しました。"
else
    echo "[INFO] GPU が見つかりません。CPU バージョンをインストールします。"
    pip install .[cpu] --quiet && echo "[INFO] CPU 依存関係のインストールが完了しました。"
fi

# ================================
# 5. Accelerate 設定
# ================================
echo "[INFO] Accelerate の設定を行います..."
accelerate config default

# ================================
# 6. デフォルトモデルのクローン
# ================================
echo "[INFO] デフォルトモデルの存在を確認します..."

if [ ! -d "$DEFAULT_MODEL_DIR" ]; then
    echo "[INFO] デフォルトモデルが見つかりません。クローンを開始します..."
    python -c "
from huggingface_hub import snapshot_download;
snapshot_download(repo_id='$DEFAULT_MODEL_REPO', local_dir='$DEFAULT_MODEL_DIR', resume_download=True)
" && echo "[INFO] モデルのクローンが完了しました: $DEFAULT_MODEL_DIR"
else
    echo "[INFO] デフォルトモデルは既に存在します: $DEFAULT_MODEL_DIR"
fi

# ================================
# 7. アプリケーションの起動
# ================================
export PYTHONWARNINGS="ignore::FutureWarning"
export CUDA_VISIBLE_DEVICES=""

echo "[INFO] アプリケーションを起動します..."
python -m med_rlhf.main

echo "[INFO] 処理が完了しました。"
