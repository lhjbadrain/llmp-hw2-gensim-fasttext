"""Read public annotation metadata and recover article bodies from local coursework data."""
import hashlib
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def article_lookup():
    path = BASE / "data" / "processed" / "1998-01.paragraphs.jsonl"
    if not path.is_file():
        raise FileNotFoundError("Run 00_clean_january.py after providing the January raw corpus.")
    groups = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        groups[row["article_id"]].append(row)
    result = {}
    for aid, rows in groups.items():
        rows.sort(key=lambda row: row["source_line"])
        result[aid] = {"text": "\n".join("".join(row["tokens"]) for row in rows), "tokens": [word for row in rows for word in row["tokens"]]}
    return result


def load_annotations(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    articles = article_lookup()
    for item in data["items"]:
        body = articles[item["article_id"]]
        expected = item["source_text_sha256"]
        if hashlib.sha256(body["text"].encode("utf-8")).hexdigest() != expected:
            raise ValueError("Annotation text hash differs: " + item["article_id"])
        item["text"] = body["text"]
        item["tokens"] = body["tokens"]
    return data
