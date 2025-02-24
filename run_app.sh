#!/usr/bin/env bash

set -e  # エラー発生時に即終了
export PYTHONWARNINGS="ignore::FutureWarning"

# ================================
# 設定
# ================================
REQUIRED_PYTHON_VERSION="3.11"
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
    sudo apt-get install -y libstdc++6
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
pip install --quiet  --upgrade pip setuptools wheel > /dev/null 2>&1 && echo "[INFO] パッケージが最新です。"
pip cache purge


# ================================
# 6. GPU/CPU 環境の確認と依存関係のインストール
# ================================
echo "[INFO] GPU 環境の有無を確認しています..."

# nvidia-smi が存在するか確認
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] nvidia-smi が見つかりました。GPU を確認中..."
    GPU_AVAILABLE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    if [ "$GPU_AVAILABLE" -ge 1 ]; then
        echo "[INFO] ✅ GPU 環境が検出されました ($GPU_AVAILABLE 台)。GPU バージョンをインストールします。"

        # GPU用の依存関係を pyproject.toml からインストール
        pip install bitsandbytes --no-build-isolation --no-cache-dir --quiet
        pip install flash-attn --no-build-isolation --no-cache-dir --quiet
        pip install triton --quiet  --no-build-isolation --no-cache-dir  --quiet && echo "[INFO] tritonのインストールが完了しました。"
        pip install --use-deprecated=legacy-resolver  --no-cache-dir .[gpu] --quiet --extra-index-url https://download.pytorch.org/whl/cu121 && echo "[INFO] ✅ GPU 依存関係のインストール完了"


    else
        echo "[WARNING] ❌ nvidia-smi は見つかりましたが、GPU は検出されませんでした。CPU バージョンをインストールします。"
        pip install --use-deprecated=legacy-resolver  .[cpu] --quiet && echo "[INFO] ✅ CPU 依存関係のインストール完了"
    fi
else
    echo "[WARNING] ❌ nvidia-smi が見つかりません。CPU バージョンをインストールします。"
    pip install --use-deprecated=legacy-resolver  .[cpu] --quiet && echo "[INFO] ✅ CPU 依存関係のインストール完了"
fi

# ================================
# 7. Accelerate の設定
# ================================
echo "[INFO] Accelerate のインストールと設定を行います..."
pip install accelerate --quiet && echo "[INFO] ✅ Accelerate のインストール完了"

# 設定ファイルの権限修正
ACCELERATE_CONFIG_DIR="${HOME}/.cache/huggingface/accelerate"
mkdir -p "$ACCELERATE_CONFIG_DIR"
chmod -R 777 "$ACCELERATE_CONFIG_DIR"

# Accelerate 設定
if command -v accelerate &> /dev/null; then
    echo "[INFO] Accelerate 設定を実行します..."
    HF_HOME="$ACCELERATE_CONFIG_DIR" accelerate config default
else
    echo "[ERROR] accelerate が見つかりません。インストールに失敗した可能性があります。"
    exit 1
fi


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
        echo "[INFO] 親ディレクトリを作成します: $DEFAULT_MODEL_DIR"
        mkdir -p "$DEFAULT_MODEL_DIR"

        # 作成成功か確認
        if [ ! -d "$DEFAULT_MODEL_DIR" ]; then
            echo "[ERROR] 親ディレクトリの作成に失敗しました: $DEFAULT_MODEL_DIR"
            exit 1
        fi

        # モデルのクローン
        python$REQUIRED_PYTHON_VERSION - <<END
from huggingface_hub import snapshot_download
import os

try:
    # 確認用ログ
    print("[INFO] snapshot_download を開始します。")
    print(f"[DEBUG] repo_id: '$DEFAULT_MODEL_REPO'")
    print(f"[DEBUG] local_dir: '$DEFAULT_MODEL_DIR'")

    # モデルのクローン
    snapshot_download(
        repo_id='$DEFAULT_MODEL_REPO', 
        local_dir='$DEFAULT_MODEL_DIR', 
        resume_download=True,
        cache_dir=os.path.expanduser("~/.cache/huggingface")  # キャッシュディレクトリ
    )

    # モデルパスの保存
    LAST_MODEL_PATH_FILE = '.last_model_path'
    with open(LAST_MODEL_PATH_FILE, 'w') as f:
        f.write('$DEFAULT_MODEL_DIR')

    print("[INFO] モデルのクローンとパスの保存が完了しました: $DEFAULT_MODEL_DIR")

except Exception as e:
    print(f"[ERROR] モデルのクローンに失敗しました: {e}")
    exit(1)
END

    else
        echo "[INFO] モデルのクローンをスキップしました。"
    fi
else
    echo "[INFO] デフォルトモデルは既に存在します: $DEFAULT_MODEL_DIR"
fi

sudo chown -R $USER /datadrive/hf_model
sudo chmod -R u+w /datadrive/hf_model



# ================================
# 9. アプリケーションの起動
# ================================
export PYTHONWARNINGS="ignore::FutureWarning"
export CUDA_VISIBLE_DEVICES=""

echo "[INFO] アプリケーションを起動します..."
python$REQUIRED_PYTHON_VERSION -m med_rlhf.main

echo "[INFO] 処理が完了しました。"
