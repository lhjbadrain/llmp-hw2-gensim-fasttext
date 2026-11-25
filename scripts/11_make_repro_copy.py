"""建立不含旧模型和结果的新副本，用于复现；原实验不变。"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DEST = BASE / "reproduce_run"


def main():
    global DEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="reproduce_run", help="本工程内一个尚不存在的复现子目录名")
    args = parser.parse_args()
    DEST = (BASE / args.name).resolve()
    if DEST.parent != BASE.resolve():
        raise ValueError("目标必须是本实验目录的直接子目录。")
    if DEST.exists():
        raise FileExistsError("复现副本已存在，不覆盖。")
    if DEST.resolve().parent != BASE.resolve():
        raise ValueError("复现副本目标超出本实验目录。")
    files = [Path("migration_manifest.json"), Path("requirements.lock.txt"),
        Path("reference/L1_02_clean_corpus.py"),
        Path("data/raw/1998-01-2003版-带音.txt"),
        Path("data/external/cnews.val.source.txt")]
    files += [Path("scripts") / file.name for file in sorted((BASE / "scripts").glob("*.py"))]
    files += [Path("data/annotations") / name for name in ["pilot_01.json", "pilot_02.json", "annotation_queue_01.json", "assistant_batch02.json", "assistant_fourclass_additions.json"]]
    for relative in files:
        if not (BASE / relative).is_file():
            raise FileNotFoundError(BASE / relative)
    checked = []
    for relative in files:
        source, target = BASE / relative, DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != source_hash:
            raise ValueError("副本校验失败：" + str(relative))
        checked.append({"path": relative.as_posix(), "sha256": source_hash})
    for name in ["results", "models"]:
        (DEST / name).mkdir(exist_ok=True)
    (DEST / "seed_files.json").write_text(json.dumps({"copied_from": str(BASE), "note": "Seed code, raw data and adopted labels only; no previous learned model, split or score copied.", "files": checked}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("新复现副本：", DEST)
    print("已核对复制文件数：", len(checked))
    print("未复制旧模型、划分或指标；可使用原项目的Python环境运行此副本的程序。")


if __name__ == "__main__":
    main()
