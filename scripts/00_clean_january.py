"""从一月原文件重建清洗数据；已有副本只核对，不覆盖。"""
import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "1998-01-2003版-带音.txt"
OUT = BASE / "data" / "processed" / "1998-01.paragraphs.jsonl"
REPORT = BASE / "results" / "january_rebuild_check.json"


def main():
    manifest = json.loads((BASE / "migration_manifest.json").read_text(encoding="utf-8"))
    expected = {row["destination"]: row["sha256"] for row in manifest["files"]}
    raw = RAW.read_bytes()
    raw_hash = hashlib.sha256(raw).hexdigest()
    if raw_hash != expected[RAW.relative_to(BASE).as_posix()]:
        raise ValueError("一月原文件校验值不符。")
    spec = importlib.util.spec_from_file_location("l1_clean_reference", BASE / "reference" / "L1_02_clean_corpus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    for number, raw_line in enumerate(raw.splitlines(), 1):
        text = raw_line.decode("gb18030", errors="strict")
        if not text.strip():
            continue
        record_id, article_id, tokens, removed = module.clean_line(text)
        if tokens:
            records.append({"id": record_id, "article_id": article_id, "source_line": number, "tokens": tokens})
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode("utf-8")
    output_hash = hashlib.sha256(payload).hexdigest()
    if output_hash != expected[OUT.relative_to(BASE).as_posix()]:
        raise ValueError("重建内容与迁移时使用的清洗版本不一致。")
    existed = OUT.exists()
    if existed:
        if hashlib.sha256(OUT.read_bytes()).hexdigest() != output_hash:
            raise ValueError("已有清洗文件与重建内容不一致，不覆盖。")
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(payload)
    assert hashlib.sha256(RAW.read_bytes()).hexdigest() == raw_hash
    result = {"source_sha256": raw_hash, "rebuilt_sha256": output_hash, "matches_migrated_cleaned_file": True, "source_unchanged": True, "paragraphs": len(records), "article_groups": len({row["article_id"] for row in records}), "encoding": "GB18030 strict", "reference_cleaner_sha256": hashlib.sha256((BASE / "reference" / "L1_02_clean_corpus.py").read_bytes()).hexdigest()}
    if not REPORT.exists():
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("一月数据重建校验通过：", len(records), "段；", result["article_groups"], "个文章组。")
    print("重建 SHA256 与本次使用的清洗副本一致；原文件未变。")
    print("已有副本仅核对。" if existed else "已生成清洗副本。")


if __name__ == "__main__":
    main()
