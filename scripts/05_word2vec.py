"""用外部新闻训练小型 Word2Vec，查看近邻；不进行主题分类。"""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import gensim
from gensim.models import Word2Vec

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "data" / "external" / "news600.v3.tokens.jsonl"
MODEL = BASE / "models" / "external_word2vec.model"
RESULT = BASE / "results" / "external_word2vec.json"
PARAMETERS = {
    "vector_size": 64,
    "window": 5,
    "min_count": 3,
    "sg": 1,
    "negative": 5,
    "hs": 0,
    "sample": 0.001,
    "epochs": 5,
    "workers": 1,
    "seed": 42,
    "sorted_vocab": 1,
}


def main():
    if MODEL.exists() or RESULT.exists():
        raise FileExistsError("Word2Vec实验结果已存在，先查看结果，不覆盖。")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError('请先在 PowerShell 执行 $env:PYTHONHASHSEED="42"，再运行本程序。')
    raw = SOURCE.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    # 每篇分词文章作为一条词序列；窗口不跨文章，会跨篇内句界。
    texts = [row["tokens"] for row in rows]
    print("开始学习词向量：", len(texts), "篇外部新闻；64维；5轮；单工作线程。", flush=True)
    started = time.perf_counter()
    model = Word2Vec(sentences=texts, **PARAMETERS)
    seconds = time.perf_counter() - started
    queries = ["球队", "基金", "经济"]
    neighbors = {}
    for word in queries:
        if word in model.wv.key_to_index:
            neighbors[word] = [{"word": near, "cosine_similarity": float(score)} for near, score in model.wv.most_similar(word, topn=5)]
        else:
            neighbors[word] = []
    result = {
        "purpose": "External news Word2Vec representation exploration; no theme labels used and no classification performed.",
        "source": str(SOURCE), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "documents": len(texts), "total_input_tokens": sum(map(len, texts)),
        "parameters": PARAMETERS, "vocabulary_size": len(model.wv),
        "build_and_train_seconds": seconds,
        "queries_and_neighbors": neighbors,
        "gensim_version": gensim.__version__, "python_version": sys.version,
        "platform": platform.platform(), "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "limitations": ["Small corpus; neighbor quality is exploratory, not a formal similarity benchmark.", "Each whole article is a training sequence; windows may cross sentence boundaries.", "min_count counts total token occurrences, unlike Dictionary no_below which counts documents.", "Fixed seed, hash seed and single worker reduce randomness; different package versions or hardware may still differ."],
    }
    model.save(str(MODEL))
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("词表大小：", len(model.wv))
    print("每个词的向量维数：", model.wv.vector_size)
    print("建词表与训练耗时（秒）：", round(seconds, 3))
    for word, items in neighbors.items():
        print(f"\n“{word}”的5个近邻（余弦相似度，不是准确率）：")
        if not items:
            print("该词不在训练词表中。")
        for item in items:
            print(item["word"], round(item["cosine_similarity"], 4))
    print("\n模型保存到：", MODEL)
    print("记录保存到：", RESULT)


if __name__ == "__main__":
    main()
