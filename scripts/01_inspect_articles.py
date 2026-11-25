"""教学步骤：合并一月段落，查看版面分布；不训练、不生成主题标签。"""
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "data" / "processed" / "1998-01.paragraphs.jsonl"

# 同一个 article_id 下的段落放进同一个列表。
articles = {}
paragraph_count = 0
with SOURCE.open(encoding="utf-8") as source:
    for line in source:
        record = json.loads(line)
        article_id = record["article_id"]
        articles.setdefault(article_id, []).append(record)
        paragraph_count += 1

# 按原始行号排列段落，编号用于分组与追踪，不混入正文。
for paragraphs in articles.values():
    paragraphs.sort(key=lambda record: record["source_line"])

page_counts = Counter(article_id.split("-")[1] for article_id in articles)
print("清洗后的段落数：", paragraph_count)
print("按编号合并得到的文章组数：", len(articles))
print("\n版面编号与文章组数（版面编号尚未映射成主题）：")
for page, count in sorted(page_counts.items()):
    print(f"  {page}: {count}")

# 06、08 仅是待观察的两个版面，不预先命名为国际、体育。
for page in ["06", "08"]:
    ids = sorted(key for key in articles if key.split("-")[1] == page)
    print(f"\n===== 版面 {page}：抽看前部、中部、后部的记录 =====")
    if not ids:
        continue
    for index in sorted({0, len(ids) // 2, len(ids) - 1}):
        article_id = ids[index]
        paragraphs = articles[article_id]
        first = " ".join(paragraphs[0]["tokens"])
        following = " ".join(word for record in paragraphs[1:] for word in record["tokens"])
        print(f"编号：{article_id}；段落数：{len(paragraphs)}")
        print("首段：", first[:150])
        print("后续正文节选：", following[:150] if following else "（无后续段落）")

print("\n注意：这只是结构与样例检查，没有划分数据或训练分类器。")
