import os

LAST_MODEL_PATH_FILE = ".last_model_path"  # 前回使用したモデルパスを記録
MODELS_BASE_DIR = "models"  # モデルのベースディレクトリ


def save_last_model_path(path):
    """前回使用したモデルパスをファイルに保存"""
    # モデルのベースディレクトリを含めたパスを保存
    full_path = os.path.join(MODELS_BASE_DIR, path) if not path.startswith(
        MODELS_BASE_DIR) else path
    with open(LAST_MODEL_PATH_FILE, "w") as f:
        f.write(full_path)


def load_last_model_path():
    """前回使用したモデルパスをファイルから読み込み"""
    if os.path.exists(LAST_MODEL_PATH_FILE):
        with open(LAST_MODEL_PATH_FILE, "r") as f:
            return f.read().strip()
    return None


def resolve_model_path(user_input):
    """ユーザー入力からフルパスを解決"""
    if not user_input:
        return None
    if not user_input.startswith(MODELS_BASE_DIR):
        return os.path.join(MODELS_BASE_DIR, user_input)
    return user_input
