"""从已采纳标签的文章中各取一篇，展示 FastText 平均文章向量。"""
import json
import re
import unicodedata
from pathlib import Path

from corpus_io import load_annotations

import numpy as np
from gensim.models import FastText

BASE = Path(__file__).resolve().parents[1]
MODEL = BASE / "models" / "external_fasttext.model"
QUEUE = BASE / "data" / "annotations" / "annotation_queue_01.json"
RESULT = BASE / "results" / "document_vector_demo.json"


def usable_tokens(tokens):
    # 人民日报保留原分词；与外部语料一致地规范全角字符、删除纯标点。
    words = [unicodedata.normalize("NFKC", word).strip() for word in tokens]
    return [word for word in words if re.search(r"[\u4e00-\u9fffA-Za-z0-9]", word)]


def main():
    if RESULT.exists():
        raise FileExistsError("演示结果已存在，先查看记录，不覆盖。")
    model = FastText.load(str(MODEL))
    articles = load_annotations(QUEUE)["items"]
    output = []
    for label in ["国际", "体育"]:
        # 标签只用于挑选演示文章；实际计算向量时只传入正文的词。
        article = next(row for row in articles if row["label"] == label and row["review_status"] in ["user_confirmed", "user_accepted_ai_suggestion"])
        words = usable_tokens(article["tokens"])
        if not words:
            raise ValueError("文章清理后没有有效词。")
        known = sum(word in model.wv.key_to_index for word in words)
        vectors = [model.wv[word] for word in words]
        # 对每个维度分别求平均：多少个词，仍得到一个64维向量。
        mean_vector = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(mean_vector))
        normalized = mean_vector / norm if norm > 0 else mean_vector
        record = {
            "sample_id": article["sample_id"], "article_id": article["article_id"],
            "label_for_display_only": label, "preview": article["text"][:100],
            "tokens": len(words), "known_token_occurrences": known,
            "oov_token_occurrences": len(words) - known,
            "known_token_fraction": known / len(words),
            "vector_dimensions": len(mean_vector),
            "mean_vector": mean_vector.tolist(), "l2_normalized_vector": normalized.tolist(),
            "word2vec_or_fasttext_labels_used_in_averaging": False,
        }
        output.append(record)
        print(f"\n===== {article['sample_id']}：{label}（仅供核对） =====")
        print("正文开头：", record["preview"])
        print("有效词数：", len(words))
        print("在 FastText 训练词表中的词次：", known)
        print("需利用子词生成向量的词次：", len(words) - known)
        print("训练词表覆盖比例：", f"{known / len(words):.1%}")
        print("文章向量维数：", len(mean_vector))
        print("平均向量前5个数：", [round(float(x), 5) for x in mean_vector[:5]])
    result = {
        "purpose": "Document vector aggregation demonstration only; no classifier trained or evaluated.",
        "aggregation": "Arithmetic mean over all usable token occurrences, including FastText OOV vectors; then optional unit L2 normalization.",
        "tokenization_note": "Peoples Daily original segmentation retained, with NFKC and punctuation filtering. External corpus used jieba; segmentation granularity may differ.",
        "limitations": "OOV vector quality can be poor. Coverage is a diagnostic, not prediction accuracy. Labels and IDs do not enter the aggregation.",
        "examples": output,
    }
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n记录保存到：", RESULT)


if __name__ == "__main__":
    main()
