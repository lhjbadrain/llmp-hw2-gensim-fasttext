"""整理外部新闻小语料：去掉类别字段、过滤、查重、分词；不训练模型。"""
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import jieba

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "external" / "cnews.val.source.txt"
OUT = BASE / "data" / "external" / "news800.tokens.jsonl"
REPORT = BASE / "results" / "external_corpus_check.json"
GROUPS = ["体育", "时政", "科技", "财经"]
PER_GROUP = 200
SEED = 42


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def comparable(text):
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", unicodedata.normalize("NFKC", text))).lower()


def main():
    if OUT.exists() or REPORT.exists():
        raise FileExistsError("输出已存在，先查看已有结果，不覆盖。")
    raw = RAW.read_bytes()
    expected = "5fe95ea140023c6f5b729adad3e83498a3489c32e61cd7776bb15b20d443025b"
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("外部原文件校验值发生变化。")

    # 人民日报仅供文本重合检查，不用于学习外部语料的词频或词表。
    people = defaultdict(list)
    people_source = BASE / "data" / "processed" / "1998-01.paragraphs.jsonl"
    for line in people_source.open(encoding="utf-8"):
        record = json.loads(line)
        people[record["article_id"]].extend(record["tokens"])
    people_texts = {comparable("".join(words)) for words in people.values()}
    people_joined = "\n".join(sorted(people_texts))

    candidates = defaultdict(list)
    skipped = Counter()
    seen = set()
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        # 第一个字段是来源类别，正文中不保留它。
        category, body = line.split("\t", 1)
        if category not in GROUPS:
            skipped["outside_selected_source_groups"] += 1
            continue
        text = normalize(body)
        chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
        if not 100 <= len(text) <= 4000 or chinese / max(len(text), 1) < 0.5:
            skipped["length_or_language"] += 1
            continue
        if re.search(r"^(组图|图集|视频|直播)", text):
            skipped["gallery_or_video"] += 1
            continue
        if category == "科技" and re.search(r"仅售|报价|促销|特价|导购|优惠|热卖|降价|超值|选购|推荐|杀价|抢购|元起|元购|元送", text[:100]):
            skipped["possible_product_promotion"] += 1
            continue
        key = comparable(text)
        if key in seen:
            skipped["duplicate_external_text"] += 1
            continue
        seen.add(key)
        if key in people_texts:
            skipped["exact_overlap_with_people"] += 1
            continue
        # 检查开头、中部、结尾等五个64字符片段，辅助发现摘录重合。
        # 这不是完整的近重复检测，不能证明不存在同事件报道或改写。
        starts = {int((len(key) - 64) * fraction) for fraction in [0, .25, .5, .75, 1]}
        if len(key) >= 64 and any(key[start:start + 64] in people_joined for start in starts):
            skipped["sampled_fragment_overlap_with_people"] += 1
            continue
        candidates[category].append((line_number, text))

    counts = {group: len(candidates[group]) for group in GROUPS}
    if min(counts.values()) < PER_GROUP:
        raise ValueError(f"过滤后不足每组{PER_GROUP}篇：{counts}；需要调整规模，不能重复凑数。")
    rng = random.Random(SEED)
    selected = []
    provenance = []
    for group in GROUPS:
        chosen = rng.sample(candidates[group], PER_GROUP)
        for line_number, text in chosen:
            external_id = f"cnews-val-{line_number:05d}"
            # 保留中文、英文及数字词；去掉纯标点。不引入外部停用词文件。
            tokens = [w.strip() for w in jieba.lcut(text) if re.search(r"[\u4e00-\u9fffA-Za-z0-9]", w)]
            selected.append({"external_id": external_id, "source_line": line_number, "text": text, "tokens": tokens})
            provenance.append({"external_id": external_id, "source_category_for_sampling_only": group})
    rng.shuffle(selected)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    report = {
        "source_file": str(RAW), "source_sha256": expected,
        "source_note": "Community mirror spiritx2023/ThuCnews, revision 9d8de9a; original institutional download unavailable during checking.",
        "purpose": "Independent Chinese representation learning corpus; not the Peoples Daily classification test set.",
        "source_split_note": "val belongs to the upstream CNews benchmark, not to this HW2 classification split.",
        "seed": SEED, "available_after_filters": counts, "selected_documents": len(selected),
        "selected_sampling_groups": {group: PER_GROUP for group in GROUPS},
        "total_tokens": sum(len(row["tokens"]) for row in selected),
        "vocabulary_size_before_gensim_filtering": len({word for row in selected for word in row["tokens"]}),
        "jieba_version": jieba.__version__, "skipped_counts": dict(skipped),
        "overlap_check": "All eligible candidates checked for normalized full-text equality and five sampled 64-character fragments against all 3132 Peoples Daily article groups. Matching candidates excluded. Not exhaustive near-duplicate checking.",
        "quality_limit": "Heuristic cleanup plus spot checks; not a fully manually cleaned corpus. Some boilerplate and promotion text may remain.",
        "selected_provenance": provenance,
        "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    OUT.write_bytes(payload.encode("utf-8"))
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("外部语料篇数：", len(selected))
    print("分词后的总词数：", report["total_tokens"])
    print("去重后的词种数：", report["vocabulary_size_before_gensim_filtering"])
    print("正文类别字段：已移除；来源类别仅用于抽样记录")
    print("人民日报重合候选排除数：", skipped["exact_overlap_with_people"] + skipped["sampled_fragment_overlap_with_people"])
    print("注意：上述重合检查不等于排除全部近重复或同事件报道。")
    print("\n前两篇预览：")
    for row in selected[:2]:
        print("正文开头：", row["text"][:100])
        print("前30个词：", row["tokens"][:30])
    print("\n保存到：", OUT)
    print("检查记录：", REPORT)


if __name__ == "__main__":
    main()
