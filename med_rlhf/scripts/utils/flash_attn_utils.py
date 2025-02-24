def apply_flash_attention(model):
    """
    FlashAttentionを使って高速化するための設定。
    実際にはモデルのAttention部分を置き換える処理が必要。
    """
    try:
        from flash_attn.models.gpt import GPTAttention
        print("[INFO] flash-attn is available. Enabling FlashAttention optimizations.")

        # モデルの各AttentionレイヤーをFlashAttentionに置き換え
        for name, module in model.named_modules():
            if isinstance(module, model.config.attention_layer_type):  # Attentionレイヤーのタイプ確認
                flash_attn_layer = GPTAttention(module.config)
                setattr(model, name, flash_attn_layer)

        model.config.use_flash_attention = True
    except ImportError:
        print("[WARNING] flash-attn not installed. Skipping FlashAttention setup.")
    except Exception as e:
        print(f"[ERROR] FlashAttentionの適用中にエラーが発生しました: {e}")
    return model


def enable_gradient_checkpointing(model):
    """
    Gradient Checkpointingを有効化してVRAMを節約する。
    """
    try:
        model.gradient_checkpointing_enable()
        print("[INFO] Gradient Checkpointing enabled.")
    except Exception as e:
        print(f"[ERROR] Gradient Checkpointingの有効化中にエラーが発生しました: {e}")
    return model
