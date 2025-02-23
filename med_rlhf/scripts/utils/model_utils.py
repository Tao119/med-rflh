# scripts/utils/model_utils.py

import os

LAST_MODEL_PATH_FILE = ".last_model_path"  # 前回使用したモデルパスを記録

def save_last_model_path(path):
    """前回使用したモデルパスをファイルに保存"""
    with open(LAST_MODEL_PATH_FILE, "w") as f:
        f.write(path)

def load_last_model_path():
    """前回使用したモデルパスをファイルから読み込み"""
    if os.path.exists(LAST_MODEL_PATH_FILE):
        with open(LAST_MODEL_PATH_FILE, "r") as f:
            return f.read().strip()
    return None
