"""逐篇阅读并人工标注；每次确认后保存，可输入 q 暂停后续跑。"""
import json
from collections import Counter
from pathlib import Path

from corpus_io import load_annotations

BASE = Path(__file__).resolve().parents[1]
QUEUE = BASE / "data" / "annotations" / "annotation_queue_01.json"
CHOICES = {"1": "国际", "2": "体育", "3": "其他", "4": "交叉"}


def save(data):
    # 先完整写入临时文件，再替换，减少写到一半丢失记录的风险。
    temporary = QUEUE.with_suffix(".tmp")
    public_data = {**data, "items": [{k: v for k, v in item.items() if k not in ["text", "tokens"]} for item in data["items"]]}
    temporary.write_text(json.dumps(public_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(QUEUE)


def main():
    data = load_annotations(QUEUE)
    print("只标主题，不训练模型。1=国际  2=体育  3=其他  4=交叉  0=跳过  q=暂停")
    print("国际指政治外交；体育可含体育组织运营；科技归其他；难分可选交叉或跳过。")
    try:
        for index, article in enumerate(data["items"], 1):
            if article["review_status"] != "unreviewed":
                continue
            print(f"\n===== {article['sample_id']}（{index}/{len(data['items'])}） =====")
            print(article["text"])
            while True:
                answer = input("\n输入 1/2/3/4，或 0 跳过、q 暂停：").strip().lower()
                if answer == "q":
                    return
                if answer in CHOICES:
                    article["label"] = CHOICES[answer]
                    article["review_status"] = "user_confirmed"
                    article["label_source"] = "用户阅读全文后在本地程序输入"
                    break
                if answer == "0":
                    article["label"] = None
                    article["review_status"] = "user_skipped"
                    break
                print("请输入 1、2、3、4、0 或 q。")
            save(data)
            print("已保存：", article["label"] or "跳过")
    except (KeyboardInterrupt, EOFError):
        print("\n已暂停，之前确认的标签已保存。")
    finally:
        confirmed = Counter(a["label"] for a in data["items"] if a["review_status"] in ["user_confirmed", "user_accepted_ai_suggestion"])
        pending = sum(a["review_status"] == "unreviewed" for a in data["items"])
        print("\n本批已确认：", dict(confirmed))
        print("本批尚未标注：", pending)
        print("记录位置：", QUEUE)


if __name__ == "__main__":
    main()
