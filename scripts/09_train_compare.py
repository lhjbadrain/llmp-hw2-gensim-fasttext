"""固定划分与逻辑回归，对比 TF-IDF 和冻结 FastText 平均文档向量。"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import gensim
import joblib
import numpy as np
import sklearn
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import normalize

BASE = Path(__file__).resolve().parents[1]
CLASSIFIER_PARAMS = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "random_state": 42}
TFIDF_PARAMS = {"min_df": 1, "max_df": 1.0, "ngram_range": (1, 1), "norm": "l2", "smooth_idf": True, "sublinear_tf": False}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def fasttext_documents(rows, model):
    vectors, diagnostics = [], []
    for row in rows:
        words = row["tokens"]
        if not words:
            raise ValueError("空文章：" + row["article_id"])
        # 只接收正文词序列；不使用标签、编号或标注理由。
        vectors.append(np.mean([model.wv[word] for word in words], axis=0))
        known = sum(word in model.wv.key_to_index for word in words)
        diagnostics.append({"article_id": row["article_id"], "tokens": len(words), "known_token_occurrences": known, "oov_token_occurrences": len(words) - known})
    matrix = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("出现非有限向量值。")
    return normalize(matrix, norm="l2"), diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["binary", "fourclass"], default="binary")
    args = parser.parse_args()
    task = args.task
    result_path = BASE / "results" / f"{task}_comparison.json"
    table_path = BASE / "results" / f"{task}_comparison.txt"
    model_dir = BASE / "models" / f"{task}_classification"
    if result_path.exists() or table_path.exists() or model_dir.exists():
        raise FileExistsError("本次对比结果已存在，先查看记录；不重复训练挑选成绩。")
    manifest_path = BASE / "results" / f"{task}_split.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_dir = BASE / "data" / "classification" / task
    for name in ["train", "test"]:
        if sha((data_dir / f"{name}.jsonl").read_bytes()) != manifest["output_sha256"][name]:
            raise ValueError("固定划分的数据已变化：" + name)
    train, test = read_jsonl(data_dir / "train.jsonl"), read_jsonl(data_dir / "test.jsonl")
    train_y = [row["label"] for row in train]
    test_y = [row["label"] for row in test]
    labels = sorted(set(train_y))
    if set(test_y) != set(labels):
        raise ValueError("训练与测试类别不一致。")
    model_dir.mkdir(parents=True)
    evaluations = []
    print(f"训练 {len(train)} 篇，测试 {len(test)} 篇；分类器均为逻辑回归。", flush=True)
    for representation in ["TF-IDF", "FastText平均向量"]:
        started = time.perf_counter()
        coverage = None
        if representation == "TF-IDF":
            vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, **TFIDF_PARAMS)
            # fit_transform 只接触训练文本；测试文本只调用 transform。
            train_x = vectorizer.fit_transform([" ".join(row["tokens"]) for row in train])
            test_x = vectorizer.transform([" ".join(row["tokens"]) for row in test])
            feature_dimensions = len(vectorizer.vocabulary_)
        else:
            # 模型已在独立新闻上训练；此处只加载，不更新词表或词向量。
            word_model = FastText.load(str(BASE / "models" / "external_fasttext.model"))
            train_x, train_coverage = fasttext_documents(train, word_model)
            test_x, test_coverage = fasttext_documents(test, word_model)
            feature_dimensions = word_model.wv.vector_size
            coverage = {"train": train_coverage, "test": test_coverage}
        feature_seconds = time.perf_counter() - started
        classifier = LogisticRegression(**CLASSIFIER_PARAMS)
        fit_started = time.perf_counter()
        classifier.fit(train_x, train_y)
        fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        predicted = classifier.predict(test_x)
        predict_seconds = time.perf_counter() - predict_started
        total_seconds = time.perf_counter() - started
        predictions = [{"article_id": row["article_id"], "sample_id": row["sample_id"], "true_label": row["label"], "predicted_label": str(prediction), "correct": row["label"] == str(prediction), "text": row["text"]} for row, prediction in zip(test, predicted)]
        evaluation = {
            "representation": representation, "feature_dimensions": feature_dimensions,
            "accuracy": float(accuracy_score(test_y, predicted)),
            "macro_f1": float(f1_score(test_y, predicted, labels=labels, average="macro", zero_division=0)),
            "classification_report": classification_report(test_y, predicted, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix_label_order": labels,
            "confusion_matrix": confusion_matrix(test_y, predicted, labels=labels).tolist(),
            "feature_seconds_including_test_transform": feature_seconds,
            "classifier_fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "feature_fit_predict_seconds": total_seconds,
            "predictions": predictions, "errors": [row for row in predictions if not row["correct"]],
            "fasttext_coverage": coverage,
        }
        if representation == "TF-IDF":
            joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, model_dir / "tfidf_logreg.joblib")
        else:
            joblib.dump({"classifier": classifier, "word_model_path": "models/external_fasttext.model", "aggregation": "mean_all_token_occurrences_then_unit_l2"}, model_dir / "fasttext_logreg.joblib")
        evaluations.append(evaluation)
        print(f"{representation}：准确率 {evaluation['accuracy']:.3f}；宏平均F1 {evaluation['macro_f1']:.3f}；特征维数 {feature_dimensions}")
    pretrained = json.loads((BASE / "results" / "external_fasttext.json").read_text(encoding="utf-8"))
    result = {
        "task": task, "train_documents": len(train), "test_documents": len(test),
        "split_manifest_sha256": sha(manifest_path.read_bytes()),
        "classifier_parameters": CLASSIFIER_PARAMS, "tfidf_parameters": TFIDF_PARAMS,
        "tfidf_implementation_note": "Classification baseline uses sklearn smoothed IDF; earlier Gensim representation demo used default log2(N/df). They are separate experiments, not identical weights.",
        "fasttext_training_before_classification_seconds": pretrained["build_and_train_seconds"],
        "fasttext_parameters": pretrained["parameters"],
        "fasttext_model_sha256": sha((BASE / "models" / "external_fasttext.model").read_bytes()),
        "timing_note": "Feature time includes train/test feature construction and, for FastText, model loading. Classifier fitting and prediction are also separated. Pipeline time excludes model saving and metric computation. FastText unsupervised pretraining is reported separately, not included in classification pipeline time.",
        "versions": {"gensim": gensim.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
        "evaluations": evaluations,
        "limitations": ["Small curated sample with AI-assisted user-accepted labels; not a representative benchmark or independently annotated gold test set.", "Held-out test evaluated for the fixed comparison; do not tune parameters using its scores.", "FastText uses additional independent unlabeled news; representation fitting data differ from TF-IDF train-only fitting, and this must be disclosed.", "Small timing differences are unstable; no repeated performance benchmark."],
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"{task}：固定训练{len(train)}篇、测试{len(test)}篇；同一种逻辑回归。", "表示 | 准确率 | 宏平均F1 | 特征构建秒 | 分类器训练秒 | 预测秒 | 特征及分类合计秒"]
    for item in evaluations:
        lines.append(f"{item['representation']} | {item['accuracy']:.4f} | {item['macro_f1']:.4f} | {item['feature_seconds_including_test_transform']:.6f} | {item['classifier_fit_seconds']:.6f} | {item['predict_seconds']:.6f} | {item['feature_fit_predict_seconds']:.6f}")
        lines.append(f"错例数：{len(item['errors'])}")
        for error in item["errors"][:3]:
            lines.append(f"  {error['article_id']}：真实{error['true_label']} → 预测{error['predicted_label']}；{error['text'][:100]}")
        if not item["errors"]:
            lines.append("  本次测试中无错例；不代表其他文章也不会出错。")
    lines.append(f"FastText外部词向量预训练单列：{pretrained['build_and_train_seconds']:.6f}秒。")
    lines.append("特征耗时包含测试转换和FastText加载；小样本耗时不宜据此判断速度优劣。")
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print("\n完整指标、预测及错例：", result_path)
    print("实验表：", table_path)


if __name__ == "__main__":
    main()
