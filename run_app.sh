#!/usr/bin/env bash

# set -e : 途中でエラーがあればスクリプトを停止
set -e

# 1) 作業ディレクトリをスクリプトのある場所に移動
#    (他の場所で実行しても問題ないようにするため)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 2) 仮想環境のディレクトリ名
VENV_DIR="venv"

# 3) もしvenvが存在しなければ作成
if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] 仮想環境 $VENV_DIR が存在しないため作成します..."
  python3 -m venv "$VENV_DIR"
fi

# 4) 仮想環境をアクティベート
echo "[INFO] 仮想環境をアクティベートします..."
# Linux/macOSの場合:
source "$VENV_DIR/bin/activate"

# Windowsの場合は run_app.bat など別のスクリプトを用意する必要があります

# 5) 依存関係をインストール
if [ -f "requirements.txt" ]; then
  echo "[INFO] pip install -r requirements.txt"
  pip install -r requirements.txt
else
  echo "[WARNING] requirements.txt が見つかりません。スキップします。"
fi

# 6) main.py を実行
echo "[INFO] main.py を起動します..."
python main.py

echo "[INFO] スクリプト終了 (仮想環境はこのシェルから抜けるまで有効のまま)"
