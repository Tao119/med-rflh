from transformers import PretrainedConfig, PreTrainedModel
import torch.nn as nn
import torch

# ================================
# ✅ カスタム Qwen2 Config の作成
# ================================
class Qwen2Config(PretrainedConfig):
    model_type = "qwen2"

    def __init__(self, vocab_size=50257, hidden_size=768, num_attention_heads=12, num_hidden_layers=12, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers

# ================================
# ✅ カスタム Qwen2 モデルの作成
# ================================
class Qwen2ForCausalLM(PreTrainedModel):
    config_class = Qwen2Config

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(0.1)  # Dropout 追加（任意）

        # モデルの重み初期化
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # 埋め込み
        hidden_states = self.embed(input_ids)

        # ドロップアウト（任意）
        hidden_states = self.dropout(hidden_states)

        # 出力層
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return {"loss": loss, "logits": logits}
