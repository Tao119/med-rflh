#!/usr/bin/env python3
# scripts/data_preparation.py

import os
import re
import json
from pathlib import Path

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"


def extract_qa_pairs(text: str):
    """
    超簡易的に 'Q:' と 'A:' を正規表現で拾うサンプル。
    実運用ではデータの構造に合わせてパースを調整してください。
    """
    pattern = r"(Q:.*?\nA:.*?)(?=\nQ:|\Z)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    qa_list = []
    for m in matches:
        lines = m.strip().split("\n")
        if len(lines) >= 2:
            q = lines[0].replace("Q:", "").strip()
            a = lines[1].replace("A:", "").strip()
            if q and a:
                qa_list.append((q, a))
    return qa_list


def prepare_sft_data(raw_dir: str, output_file: str):
    """
    raw_dir以下のtxtファイルを走査し、Q&Aを抽出してjsonlにまとめる。
    フィールド: {instruction, input, output}
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    txt_files = sorted(Path(raw_dir).glob("*.txt"))
    with open(output_file, "w", encoding="utf-8") as fout:
        for txtf in txt_files:
            with open(txtf, "r", encoding="utf-8") as fin:
                text = fin.read()
                qa_pairs = extract_qa_pairs(text)
                for (question, answer) in qa_pairs:
                    data_item = {
                        "instruction": question,
                        "input": "",   # 追加情報がある場合に入れる
                        "output": answer
                    }
                    fout.write(json.dumps(data_item, ensure_ascii=False))
                    fout.write("\n")


def main():
    output_file = os.path.join(PROCESSED_DATA_DIR, "sft_dataset.jsonl")
    prepare_sft_data(RAW_DATA_DIR, output_file)
    print(f"SFT用データを生成しました: {output_file}")


if __name__ == "__main__":
    main()
