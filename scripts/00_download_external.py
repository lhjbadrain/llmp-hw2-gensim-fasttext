"""获取本次独立表示学习语料；校验固定版本，已有正确副本直接复用。"""
import hashlib
import json
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "external" / "cnews.val.source.txt"
META = OUT.parent / "cnews_download.json"
URL = "https://huggingface.co/datasets/spiritx2023/ThuCnews/resolve/9d8de9a/cnews.val.txt"
EXPECTED = "5fe95ea140023c6f5b729adad3e83498a3489c32e61cd7776bb15b20d443025b"


def main():
    if OUT.exists():
        data = OUT.read_bytes()
    else:
        request = urllib.request.Request(URL, headers={"User-Agent": "HW2-small-corpus-learning/1.0"})
        with urllib.request.urlopen(request, timeout=25) as response:
            data = response.read(15 * 1024 * 1024 + 1)
        if len(data) > 15 * 1024 * 1024:
            raise ValueError("超过15MiB下载上限。")
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED:
        raise ValueError("外部文件校验值不符，不写入或覆盖。")
    data.decode("utf-8-sig", errors="strict")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not META.exists():
        META.write_text(json.dumps({"url": URL, "sha256": digest, "bytes": len(data), "provenance": "Community mirror spiritx2023/ThuCnews, fixed revision9d8de9a; not an institutional download.", "purpose": "External representation learning only; upstream category fields removed in preparation."}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("外部语料校验通过：", len(data), "字节；固定版本 SHA256 一致。")


if __name__ == "__main__":
    main()
