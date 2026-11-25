"""整理已采纳的主题标签，固定二分类划分；不训练或评估模型。"""
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from corpus_io import load_annotations

BASE = Path(__file__).resolve().parents[1]
ANNOTATIONS = BASE / "data" / "annotations"
SOURCE = BASE / "data" / "processed" / "1998-01.paragraphs.jsonl"
OUT = BASE / "data" / "classification" / "binary"
MANIFEST = BASE / "results" / "binary_split.json"
FILES = ["pilot_01.json", "pilot_02.json", "annotation_queue_01.json", "assistant_batch02.json"]
LABELS = ["国际", "体育"]
SEED = 42
TEST_PER_CLASS = 5


def digest(data):
    return hashlib.sha256(data).hexdigest()


def clean_tokens(tokens):
    words = [unicodedata.normalize("NFKC", word).strip() for word in tokens]
    return [word for word in words if re.search(r"[\u4e00-\u9fffA-Za-z0-9]", word)]


def main():
    if OUT.exists() or MANIFEST.exists():
        raise FileExistsError("划分记录已存在，先查看，不覆盖或重新挑选测试集。")
    paragraphs = defaultdict(list)
    for line in SOURCE.open(encoding="utf-8"):
        row = json.loads(line)
        paragraphs[row["article_id"]].append(row)
    rows, early_ids, annotation_hashes = [], set(), {}
    for filename in FILES:
        raw = (ANNOTATIONS / filename).read_bytes()
        annotation_hashes[filename] = digest(raw)
        document = load_annotations(ANNOTATIONS / filename)
        for annotation in document["items"]:
            if annotation.get("label") not in LABELS:
                continue
            if annotation["review_status"] not in ["user_confirmed", "user_accepted_ai_suggestion"]:
                raise ValueError("有主题标签尚未采纳：" + annotation["sample_id"])
            aid = annotation["article_id"]
            article_paragraphs = sorted(paragraphs[aid], key=lambda record: record["source_line"])
            text = "\n".join("".join(record["tokens"]) for record in article_paragraphs)
            tokens = clean_tokens([word for record in article_paragraphs for word in record["tokens"]])
            if not tokens or text != annotation["text"]:
                raise ValueError("正文为空或与标注来源不一致：" + aid)
            normalized = "".join(tokens).lower()
            rows.append({
                "article_id": aid, "sample_id": annotation["sample_id"],
                "label": annotation["label"], "text": text, "tokens": tokens,
                "event_group": annotation.get("event_group", aid),
                "normalized_text_sha256": digest(normalized.encode("utf-8")),
                "annotation_file": filename, "label_source": annotation["label_source"] if "label_source" in annotation else "用户在聊天确认",
                "annotation_status": annotation["review_status"],
            })
            if filename != "assistant_batch02.json":
                early_ids.add(aid)
    rows.sort(key=lambda record: record["article_id"])
    by_id = {record["article_id"]: record for record in rows}
    if len(by_id) != len(rows) or Counter(record["label"] for record in rows) != Counter({"国际": 20, "体育": 20}):
        raise ValueError("预期40篇不重复文章，每类20篇，实际不符。")

    # 合并同事件或相同正文的分组，防止一组文章出现在划分两侧。
    parent = {aid: aid for aid in by_id}
    def find(aid):
        while parent[aid] != aid:
            parent[aid] = parent[parent[aid]]
            aid = parent[aid]
        return aid
    def union(first, second):
        parent[find(second)] = find(first)
    seen_event, seen_text = {}, {}
    for record in rows:
        aid = record["article_id"]
        for seen, key in [(seen_event, record["event_group"]), (seen_text, record["normalized_text_sha256"])]:
            if key in seen:
                union(aid, seen[key])
            else:
                seen[key] = aid
    for record in rows:
        record["split_group"] = find(record["article_id"])
    early_groups = {find(aid) for aid in early_ids}
    eligible = {label: [record["article_id"] for record in rows if record["label"] == label and record["split_group"] not in early_groups] for label in LABELS}
    rng = random.Random(SEED)
    for attempt in range(1, 1001):
        test_ids = {aid for label in LABELS for aid in rng.sample(eligible[label], TEST_PER_CLASS)}
        test_groups = {find(aid) for aid in test_ids}
        train_groups = {find(aid) for aid in by_id if aid not in test_ids}
        if test_groups.isdisjoint(train_groups):
            break
    else:
        raise ValueError("未找到满足组不重合的固定划分。")
    train = [record for record in rows if record["article_id"] not in test_ids]
    test = [record for record in rows if record["article_id"] in test_ids]
    train_hashes = {record["normalized_text_sha256"] for record in train}
    test_hashes = {record["normalized_text_sha256"] for record in test}
    assert train_hashes.isdisjoint(test_hashes)
    assert early_ids.isdisjoint(test_ids)
    assert Counter(record["label"] for record in train) == Counter({"国际": 15, "体育": 15})
    assert Counter(record["label"] for record in test) == Counter({"国际": 5, "体育": 5})
    OUT.mkdir(parents=True)
    output_hashes = {}
    for name, records in [("train", train), ("test", test)]:
        payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode("utf-8")
        (OUT / f"{name}.jsonl").write_bytes(payload)
        output_hashes[name] = digest(payload)
    manifest = {
        "purpose": "Fixed small curated Peoples Daily theme classification split; not a representative January benchmark.",
        "seed": SEED, "sampling_attempts_until_group_constraints_met": attempt,
        "split_rule": "Per-class fixed five test examples from the newly adopted batch. Reject group-overlapping draws without fitting or scoring any model. Earlier rule-discussion and vector-demo samples forced into training.",
        "train_counts": dict(Counter(record["label"] for record in train)),
        "test_counts": dict(Counter(record["label"] for record in test)),
        "train_article_ids": [record["article_id"] for record in train],
        "test_article_ids": [record["article_id"] for record in test],
        "forced_training_ids": sorted(early_ids),
        "checks": {"article_id_overlap": 0, "normalized_text_overlap": 0, "identified_group_overlap": 0, "early_calibration_in_test": 0},
        "feature_policy": "Only normalized正文tokens used for features. Label, ID, page, event group, annotation rationale and source file are metadata only. Future TF-IDF fits only train tokens; frozen external FastText does not train on this test data.",
        "source_sha256": digest(SOURCE.read_bytes()), "annotation_sha256": annotation_hashes,
        "output_sha256": output_hashes,
        "limitations": ["40 curated articles, 10 test articles; one error changes accuracy by 10 percentage points.", "Themes partly AI-assisted and accepted by user, not independently double-annotated gold labels.", "Known groups and exact normalized duplicates controlled, not exhaustive near-duplicate or same-event discovery."],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("主题标签：国际、体育（来自全文标注，不是版面标签）")
    print("训练集：", dict(Counter(record["label"] for record in train)), "合计", len(train))
    print("测试集：", dict(Counter(record["label"] for record in test)), "合计", len(test))
    print("此前用于规则讨论和向量演示的样本：全部留在训练集")
    print("文章编号重合：0；规范化正文重合：0；已识别事件组重合：0")
    print("固定随机种子：", SEED)
    print("数据目录：", OUT)
    print("划分记录：", MANIFEST)
    print("提醒：测试集仅10篇，一篇错误会使准确率变化10个百分点。")


if __name__ == "__main__":
    main()
