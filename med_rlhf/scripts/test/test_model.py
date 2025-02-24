import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model():
    model_id = "rinna/deepseek-r1-distill-qwen2.5-bakeneko-32b"
    print(f"[INFO] モデル {model_id} をロード中...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print("[INFO] モデルのロード完了！")
    print("[INFO] 対話を開始します。（終了するには 'exit' または 'quit' と入力してください）")
    
    # 対話ループ
    while True:
        user_input = input("User> ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("[INFO] 対話を終了します。")
            break
        
        # チャット形式のメッセージリストを作成（ユーザーからの入力のみ）
        messages = [{"role": "user", "content": user_input}]
        
        # トークナイザーの apply_chat_template メソッドを利用してプロンプトを生成
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # プロンプトをエンコード
        input_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
        
        # 応答の生成
        outputs = model.generate(
            input_ids,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
        )
        
        # プロンプト部分を除いた生成部分をデコード
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        # 必要に応じて出力前にテキストを挿入
        response = "<think>\n" + response
        print("Bot> " + response)
