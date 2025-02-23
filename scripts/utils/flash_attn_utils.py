# scripts/utils/flash_attn_utils.py

def apply_flash_attention(model):
    """
    xFormers or flash-attnを使って高速化するための簡易フラグ設定。
    実際にはモデルのAttention部分を置き換える処理が必要になる場合も。
    """
    try:
        import xformers
        print("[INFO] xformers is available. Enabling xformers optimizations.")
        model.config.use_xformers = True
    except ImportError:
        print("[WARNING] xformers not installed. Skipping FlashAttention setup.")
    return model


def enable_gradient_checkpointing(model):
    """
    Gradient Checkpointingを有効化してVRAMを節約する。
    """
    model.gradient_checkpointing_enable()
    return model
