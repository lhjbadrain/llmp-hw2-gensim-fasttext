"""Gensim FastText 词向量与未登录词实验；不是监督分类程序。"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import gensim
from gensim.models import FastText, Word2Vec

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "data" / "external" / "news600.v3.tokens.jsonl"
MODEL = BASE / "models" / "external_fasttext.model"
RESULT = BASE / "results" / "external_fasttext.json"
PARAMETERS = {
    "vector_size": 64, "window": 5, "min_count": 3,
    "sg": 1, "negative": 5, "hs": 0, "sample": 0.001,
    "epochs": 5, "workers": 1, "seed": 42, "sorted_vocab": 1,
    "min_n": 2, "max_n": 4, "bucket": 20000,
}


def main():
    if MODEL.exists() or RESULT.exists():
        raise FileExistsError("FastText实验结果已存在，先查看记录，不覆盖。")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError('请先在 PowerShell 执行 $env:PYTHONHASHSEED="42"。')
    raw = SOURCE.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    texts = [row["tokens"] for row in rows]
    previous = json.loads((BASE / "results" / "external_word2vec.json").read_text(encoding="utf-8"))
    if previous["source_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("Word2Vec与本次语料不一致，停止直接比较。")
    for key in previous["parameters"]:
        if previous["parameters"][key] != PARAMETERS[key]:
            raise ValueError("共有参数不一致：" + key)
    print("开始 FastText 词向量训练：600篇新闻；64维；5轮；子词长度2—4。", flush=True)
    started = time.perf_counter()
    model = FastText(sentences=texts, **PARAMETERS)
    seconds = time.perf_counter() - started
    neighbors = {}
    for word in ["球队", "基金", "经济"]:
        neighbors[word] = [{"word": near, "cosine_similarity": float(score)} for near, score in model.wv.most_similar(word, topn=5)] if word in model.wv.key_to_index else []

    # 同时检查训练文本，确保示例确实没有作为完整词出现过。
    seen = {word for words in texts for word in words}
    candidate_words = ["球队管理员", "基金投资者", "篮球爱好者俱乐部"]
    oov_word = next((word for word in candidate_words if word not in seen and word not in model.wv.key_to_index), None)
    oov = None
    if oov_word is not None:
        old_model = Word2Vec.load(str(BASE / "models" / "external_word2vec.model"))
        try:
            old_model.wv[oov_word]
            w2v_lookup = "vector_available"
        except KeyError:
            w2v_lookup = "KeyError: word not in Word2Vec vocabulary"
        vector = model.wv[oov_word]
        oov = {
            "word": oov_word, "appeared_as_whole_token_in_training": False,
            "in_fasttext_word_vocabulary": oov_word in model.wv.key_to_index,
            "word2vec_lookup": w2v_lookup, "fasttext_vector_dimensions": len(vector),
            "fasttext_vector_first5": [float(x) for x in vector[:5]],
            "fasttext_neighbors": [{"word": near, "cosine_similarity": float(score)} for near, score in model.wv.most_similar(oov_word, topn=5)],
        }
    result = {
        "purpose": "Gensim unsupervised FastText word vectors and subword OOV experiment; not official fastText supervised classification.",
        "source_sha256": hashlib.sha256(raw).hexdigest(), "documents": len(texts),
        "parameters": PARAMETERS, "vocabulary_size": len(model.wv),
        "build_and_train_seconds": seconds, "queries_and_neighbors": neighbors, "oov_example": oov,
        "gensim_version": gensim.__version__, "python_version": sys.version,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "limitations": ["Small corpus and reduced bucket size; hash collisions and limited contexts may affect quality.", "OOV vector availability does not prove semantic accuracy.", "Neighbor comparisons are exploratory; similarity numbers are not classification accuracy.", "Whole articles are training sequences; context windows may cross sentence boundaries."],
    }
    model.save(str(MODEL))
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("词表大小：", len(model.wv))
    print("建词表与训练耗时（秒）：", round(seconds, 3))
    for word, items in neighbors.items():
        print(f"\n“{word}”的5个近邻：")
        for item in items:
            print(item["word"], round(item["cosine_similarity"], 4))
    if oov:
        print("\n未登录词实验：", oov["word"])
        print("在训练中作为完整词出现过：", oov["appeared_as_whole_token_in_training"])
        print("Word2Vec 查询结果：", oov["word2vec_lookup"])
        print("在 FastText 训练词表中：", oov["in_fasttext_word_vocabulary"])
        print("FastText 生成向量的维数：", oov["fasttext_vector_dimensions"])
        print("该词的5个近邻：")
        for item in oov["fasttext_neighbors"]:
            print(item["word"], round(item["cosine_similarity"], 4))
    print("\n模型保存到：", MODEL)
    print("记录保存到：", RESULT)


if __name__ == "__main__":
    main()
