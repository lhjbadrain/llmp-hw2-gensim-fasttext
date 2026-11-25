"""四分类固定划分：保留原二分类划分，新增经济/科技各15训练、5测试。"""
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from corpus_io import load_annotations

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "data" / "processed" / "1998-01.paragraphs.jsonl"
ANNOTATIONS = BASE / "data" / "annotations" / "assistant_fourclass_additions.json"
BINARY = BASE / "data" / "classification" / "binary"
OUT = BASE / "data" / "classification" / "fourclass"
MANIFEST = BASE / "results" / "fourclass_split.json"
LABELS = ["国际", "体育", "经济", "科技"]
SEED = 42


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def clean_tokens(tokens):
    words = [unicodedata.normalize("NFKC", word).strip() for word in tokens]
    return [word for word in words if re.search(r"[\u4e00-\u9fffA-Za-z0-9]", word)]


def main():
    if OUT.exists() or MANIFEST.exists():
        raise FileExistsError("四分类划分已存在，先查看，不覆盖。")
    binary_manifest_path = BASE / "results" / "binary_split.json"
    binary_manifest = json.loads(binary_manifest_path.read_text(encoding="utf-8"))
    for part in ["train", "test"]:
        if digest((BINARY / f"{part}.jsonl").read_bytes()) != binary_manifest["output_sha256"][part]:
            raise ValueError("原二分类划分已变化。")
    old_train, old_test = read_rows(BINARY / "train.jsonl"), read_rows(BINARY / "test.jsonl")
    old_train_ids = {row["article_id"] for row in old_train}
    old_test_ids = {row["article_id"] for row in old_test}
    paragraphs = defaultdict(list)
    for row in read_rows(SOURCE):
        paragraphs[row["article_id"]].append(row)
    annotations = load_annotations(ANNOTATIONS)
    additions, forced_ids = [], set()
    for annotation in annotations["items"]:
        if annotation["review_status"] != "user_accepted_ai_suggestion":
            raise ValueError("还有未采纳的标签。")
        aid = annotation["article_id"]
        records = sorted(paragraphs[aid], key=lambda row: row["source_line"])
        text = "\n".join("".join(row["tokens"]) for row in records)
        words = clean_tokens([word for row in records for word in row["tokens"]])
        if not words or text != annotation["text"]:
            raise ValueError("标注与原文不一致：" + aid)
        additions.append({
            "article_id": aid, "sample_id": annotation["sample_id"], "label": annotation["label"],
            "text": text, "tokens": words, "event_group": annotation.get("event_group", aid),
            "normalized_text_sha256": digest("".join(words).lower().encode("utf-8")),
            "annotation_file": ANNOTATIONS.name, "label_source": annotation["label_source"],
            "annotation_status": annotation["review_status"],
        })
        if annotation["force_training"]:
            forced_ids.add(aid)
    if Counter(row["label"] for row in additions) != Counter({"经济": 20, "科技": 20}):
        raise ValueError("新增类别数量不符。")
    rows = sorted(old_train + old_test + additions, key=lambda row: row["article_id"])
    by_id = {row["article_id"]: row for row in rows}
    if len(rows) != 80 or len(by_id) != 80:
        raise ValueError("预期80篇互不重复文章。")
    parent = {aid: aid for aid in by_id}
    def find(aid):
        while parent[aid] != aid:
            parent[aid] = parent[parent[aid]]
            aid = parent[aid]
        return aid
    def union(first, second):
        parent[find(second)] = find(first)
    seen_events, seen_texts = {}, {}
    for row in rows:
        for seen, key in [(seen_events, row["event_group"]), (seen_texts, row["normalized_text_sha256"])]:
            if key in seen:
                union(row["article_id"], seen[key])
            else:
                seen[key] = row["article_id"]
    force_train_groups = {find(aid) for aid in forced_ids | old_train_ids}
    fixed_test_groups = {find(aid) for aid in old_test_ids}
    if force_train_groups & fixed_test_groups:
        raise ValueError("新发现文章组与旧划分冲突，需检查；不能静默改动旧划分。")
    eligible = {label: [row["article_id"] for row in additions if row["label"] == label and find(row["article_id"]) not in force_train_groups] for label in ["经济", "科技"]}
    rng = random.Random(SEED)
    for attempt in range(1, 1001):
        new_test_ids = {aid for label in ["经济", "科技"] for aid in rng.sample(eligible[label], 5)}
        test_ids = old_test_ids | new_test_ids
        test_groups = {find(aid) for aid in test_ids}
        train_groups = {find(aid) for aid in by_id if aid not in test_ids}
        if not test_groups & train_groups:
            break
    else:
        raise ValueError("未找到满足分组约束的固定划分。")
    train = [row for row in rows if row["article_id"] not in test_ids]
    test = [row for row in rows if row["article_id"] in test_ids]
    for row in rows:
        row["split_group"] = find(row["article_id"])
    assert Counter(row["label"] for row in train) == Counter({label: 15 for label in LABELS})
    assert Counter(row["label"] for row in test) == Counter({label: 5 for label in LABELS})
    assert old_train_ids <= {row["article_id"] for row in train}
    assert old_test_ids <= test_ids
    assert not forced_ids & test_ids
    assert {row["normalized_text_sha256"] for row in train}.isdisjoint(row["normalized_text_sha256"] for row in test)
    OUT.mkdir(parents=True)
    output_hashes = {}
    for part, records in [("train", train), ("test", test)]:
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode("utf-8")
        (OUT / f"{part}.jsonl").write_bytes(payload)
        output_hashes[part] = digest(payload)
    manifest = {
        "purpose": "Four-theme small curated classification split, not a representative January benchmark.",
        "seed": SEED, "sampling_attempts_until_group_constraints_met": attempt,
        "train_counts": dict(Counter(row["label"] for row in train)),
        "test_counts": dict(Counter(row["label"] for row in test)),
        "train_article_ids": [row["article_id"] for row in train],
        "test_article_ids": [row["article_id"] for row in test],
        "forced_training_ids": sorted(forced_ids),
        "preserves_binary_split": True,
        "checks": {"article_id_overlap": 0, "normalized_text_overlap": 0, "identified_group_overlap": 0, "earlier_refinement_in_test": 0},
        "source_sha256": digest(SOURCE.read_bytes()),
        "annotation_sha256": digest(ANNOTATIONS.read_bytes()),
        "binary_manifest_sha256": digest(binary_manifest_path.read_bytes()),
        "output_sha256": output_hashes,
        "split_rule": "Retain binary train/test membership, select five new test articles per new class with seed42, force prior refinements and identified overlapping technology-review group to train. Reject group-overlapping draws without using model scores.",
        "feature_policy": "Use tokens only. Labels, IDs, pages, event groups and annotation reasons are metadata. TF-IDF fits train only; FastText stays frozen from external corpus.",
        "limitations": ["80 curated articles, 20 test articles; one test mistake changes accuracy by five percentage points.", "AI-assisted user-accepted labels, not independently double-annotated gold data.", "Known event groups and exact normalized duplicates controlled; exhaustive near-duplicate checks not performed.", "Old binary test articles reused in expanded four-class task; this is a nested comparison, not an independent replication.", "Frozen external corpus contains sports/politics/finance sampling groups; technology coverage may be limited."],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("四分类主题：国际、体育、经济、科技")
    print("训练集：", manifest["train_counts"], "合计", len(train))
    print("测试集：", manifest["test_counts"], "合计", len(test))
    print("原二分类训练/测试归属：全部保留")
    print("前期已讨论的细分样例：全部留在训练集")
    print("文章编号重合：0；规范化正文重合：0；已识别事件组重合：0")
    print("数据目录：", OUT)
    print("划分记录：", MANIFEST)
    print("下一步运行：09_train_compare.py --task fourclass")


if __name__ == "__main__":
    main()
