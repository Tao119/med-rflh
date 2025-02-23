#!/usr/bin/env bash

set -e  # エラー発生時に即終了
export PYTHONWARNINGS="ignore::FutureWarning"

# ================================
# 設定
# ================================
REQUIRED_PYTHON_VERSION="3.12"
VENV_DIR=".venv"
DEFAULT_MODEL_REPO="rinna/deepseek-r1-distill-qwen2.5-bakeneko-32b"
DEFAULT_MODEL_DIR="models/base/rinna-deepseek-r1-distill-qwen2.5-bakeneko-32b"


# ================================
# 1. OS検出
# ================================
OS_TYPE="$(uname)"
echo "[INFO] OS タイプ検出: $OS_TYPE"

if [[ "$OS_TYPE" == "Darwin" ]]; then
    PLATFORM="macOS"
elif [[ "$OS_TYPE" == "Linux" ]]; then
    PLATFORM="Linux"
elif [[ "$OS_TYPE" == MINGW* || "$OS_TYPE" == CYGWIN* || "$OS_TYPE" == MSYS* ]]; then
    PLATFORM="Windows"
else
    echo "[ERROR] 未対応のOSです: $OS_TYPE"
    exit 1
fi

echo "[INFO] 検出されたプラットフォーム: $PLATFORM"

# ================================
# 2. システム依存関係の確認とインストール
# ================================
echo "[INFO] システム依存関係を確認しています..."

# 必要なパッケージのリスト (Linux/macOS用)
DEPENDENCIES=("cmake" "pkg-config" "protobuf-compiler" "python$REQUIRED_PYTHON_VERSION" "python$REQUIRED_PYTHON_VERSION-venv" "python$REQUIRED_PYTHON_VERSION-dev")

# インストール関数
install_dependencies_linux() {
    echo "[INFO] (Linux) 以下の依存関係をインストールします: ${DEPENDENCIES[*]}"
    sudo apt update
    sudo apt install -y "${DEPENDENCIES[@]}"
}

install_dependencies_mac() {
    echo "[INFO] (macOS) 必要なパッケージをHomebrewでインストールします..."
    brew update
    brew install cmake protobuf python@$REQUIRED_PYTHON_VERSION
}

# 依存関係の確認とインストール
if [[ "$PLATFORM" == "Linux" ]]; then
    for pkg in "${DEPENDENCIES[@]}"; do
        if ! dpkg -s "$pkg" &>/dev/null; then
            echo "[WARNING] $pkg が見つかりません。"
            install_dependencies_linux
            break
        fi
    done
elif [[ "$PLATFORM" == "macOS" ]]; then
    if ! command -v brew &> /dev/null; then
        echo "[ERROR] Homebrew が見つかりません。インストールしてください: https://brew.sh/"
        exit 1
    fi
    install_dependencies_mac
elif [[ "$PLATFORM" == "Windows" ]]; then
    echo "[WARNING] Windows 環境では依存関係を手動でインストールしてください。"
fi

# ================================
# 3. Python バージョンの確認
# ================================
echo "[INFO] Python バージョンを確認しています..."

if ! command -v python$REQUIRED_PYTHON_VERSION &> /dev/null; then
    echo "[WARNING] Python $REQUIRED_PYTHON_VERSION が見つかりません。"

    if [[ "$PLATFORM" == "Linux" ]]; then
        echo "[INFO] (Linux) Python をインストールします..."
        sudo apt install -y python$REQUIRED_PYTHON_VERSION python$REQUIRED_PYTHON_VERSION-venv python$REQUIRED_PYTHON_VERSION-dev
    elif [[ "$PLATFORM" == "macOS" ]]; then
        echo "[INFO] (macOS) HomebrewでPythonをインストールします..."
        brew install python@$REQUIRED_PYTHON_VERSION
    elif [[ "$PLATFORM" == "Windows" ]]; then
        echo "[ERROR] Windows では手動で Python をインストールしてください: https://www.python.org/downloads/"
        exit 1
    fi
fi

# バージョン確認
CURRENT_PYTHON_VERSION=$(python$REQUIRED_PYTHON_VERSION --version 2>&1 | awk '{print $2}')
echo "[INFO] 使用する Python バージョン: $CURRENT_PYTHON_VERSION"

# ================================
# 4. 仮想環境の作成
# ================================
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] 仮想環境を作成します: $VENV_DIR"
    python$REQUIRED_PYTHON_VERSION -m venv "$VENV_DIR"
else
    echo "[INFO] 仮想環境は既に存在します: $VENV_DIR"
fi

# 仮想環境のアクティベート
if [[ "$PLATFORM" == "Windows" ]]; then
    source "$VENV_DIR/Scripts/activate"
else
    source "$VENV_DIR/bin/activate"
fi

# ================================
# 5. 必要なPythonパッケージのインストール
# ================================
echo "[INFO] pip, setuptools, wheel をアップグレードします..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1 && echo "[INFO] パッケージが最新です。"

# ================================
# 6. GPU/CPU 環境の確認と依存関係のインストール
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
# 7. Accelerate 設定
# ================================
echo "[INFO] Accelerate の設定を行います..."
accelerate config default

# ================================
# 8. デフォルトモデルのクローン
# ================================
echo "[INFO] デフォルトモデルの存在を確認します..."

if [ ! -d "$DEFAULT_MODEL_DIR" ]; then
    echo "[INFO] デフォルトモデルが見つかりません。"

    # ユーザーに確認
    read -p "デフォルトモデルをクローンしますか？ (y/n): " user_input

    if [[ "$user_input" =~ ^[Yy]$ ]]; then
        echo "[INFO] クローンを開始します..."

        # 親ディレクトリを作成
        mkdir -p "$(dirname "$DEFAULT_MODEL_DIR")"

        # モデルのクローン
        python -c "
from huggingface_hub import snapshot_download

# モデルのクローン
snapshot_download(repo_id='$DEFAULT_MODEL_REPO', local_dir='$DEFAULT_MODEL_DIR', resume_download=True)

# モデルパスの保存
LAST_MODEL_PATH_FILE = '.last_model_path'
with open(LAST_MODEL_PATH_FILE, 'w') as f:
    f.write('$DEFAULT_MODEL_DIR')
" && echo "[INFO] モデルのクローンとパスの保存が完了しました: $DEFAULT_MODEL_DIR"
    else
        echo "[INFO] モデルのクローンをスキップしました。"
    fi
else
    echo "[INFO] デフォルトモデルは既に存在します: $DEFAULT_MODEL_DIR"
fi

# ================================
# 9. アプリケーションの起動
# ================================
export PYTHONWARNINGS="ignore::FutureWarning"
export CUDA_VISIBLE_DEVICES=""

echo "[INFO] アプリケーションを起動します..."
python -m med_rlhf.main

echo "[INFO] 処理が完了しました。"
