"""Gensim 表示学习第一步：词袋计数与 TF-IDF。仅使用外部新闻。"""
import hashlib
import json
import time
from pathlib import Path

import gensim
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "data" / "external" / "news600.v3.tokens.jsonl"
MODEL_DIR = BASE / "models" / "external_bow_tfidf"
RESULT = BASE / "results" / "external_bow_tfidf.json"


def main():
    if MODEL_DIR.exists() or RESULT.exists():
        raise FileExistsError("本实验结果已存在，先查看结果，不覆盖。")
    started = time.perf_counter()
    payload = SOURCE.read_bytes()
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    texts = [row["tokens"] for row in rows]

    # 词典把每个词对应到一个整数编号。这里仅在外部新闻上拟合。
    dictionary = Dictionary(texts)
    before = len(dictionary)
    dictionary.filter_extremes(no_below=3, no_above=0.8, keep_n=20000)
    # BoW 中的每一项是 (词编号, 这篇文章里的出现次数)。
    corpus = [dictionary.doc2bow(words) for words in texts]
    # 用整个外部语料的文档频率计算 IDF；输出默认作 L2 归一化。
    model = TfidfModel(corpus, normalize=True)

    # 选第一篇非空文章，保证两种表示比较的是同一篇文章。
    index = next(i for i, bow in enumerate(corpus) if bow)
    bow = corpus[index]
    vector = list(model[bow])
    counts = dict(bow)
    bow_top = sorted(bow, key=lambda pair: (-pair[1], pair[0]))[:8]
    tfidf_top = sorted(vector, key=lambda pair: (-pair[1], pair[0]))[:8]
    result = {
        "purpose": "External corpus representation demonstration; no classification or test-set evaluation.",
        "gensim_version": gensim.__version__, "source_sha256": hashlib.sha256(payload).hexdigest(),
        "documents": len(rows), "dictionary_size_before": before, "dictionary_size_after": len(dictionary),
        "dictionary_filter": {"no_below": 3, "no_above": 0.8, "keep_n": 20000},
        "empty_documents_after_filtering": sum(not document for document in corpus),
        "example_id": rows[index]["external_id"], "example_preview": rows[index]["text"][:160],
        "bow_top": [{"word": dictionary[word_id], "count": count} for word_id, count in bow_top],
        "tfidf_top": [{"word": dictionary[word_id], "weight": float(weight), "count": counts[word_id], "document_frequency": dictionary.dfs[word_id]} for word_id, weight in tfidf_top],
        "tfidf_note": "Gensim default local raw term count and global log2(N/df), followed by unit L2 normalization.",
    }
    MODEL_DIR.mkdir(parents=True)
    dictionary.save(str(MODEL_DIR / "external.dictionary"))
    model.save(str(MODEL_DIR / "external.tfidf"))
    result["elapsed_seconds"] = time.perf_counter() - started
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("文档数：", len(rows))
    print("词典大小（过滤前 → 过滤后）：", before, "→", len(dictionary))
    print("过滤后空文档数：", result["empty_documents_after_filtering"])
    print("\n同一篇文章开头：", result["example_preview"])
    print("\nBoW 排名前8的词（词语、次数）：")
    for item in result["bow_top"]:
        print(item["word"], item["count"])
    print("\nTF-IDF 排名前8的词（词语、权重、本篇次数、出现文档数）：")
    for item in result["tfidf_top"]:
        print(item["word"], round(item["weight"], 4), item["count"], item["document_frequency"])
    print("\n耗时（秒）：", round(result["elapsed_seconds"], 3))
    print("结果保存到：", RESULT)


if __name__ == "__main__":
    main()
