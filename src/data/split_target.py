import argparse
import json
import random
from collections import Counter
from pathlib import Path

BUDGET = 1_000_000
DEV_RATIO = 0.2
SEED = 42


def assign(docs):
    # documents in seeded random order, each into train, else dev, else test
    order = sorted(docs, key=lambda d: d["doc"])
    random.Random(SEED).shuffle(order)
    total, where = Counter(), {}
    for d in order:
        # fill train up to its share of the budget, then dev up to its share, rest is test
        if total["train"] + d["words"] <= BUDGET * (1 - DEV_RATIO):
            split = "train"
        elif total["dev"] + d["words"] <= BUDGET * DEV_RATIO:
            split = "dev"
        else:
            split = "test"
        where[d["doc"]] = split
        total[split] += d["words"]
    return where


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="books")
    p.add_argument("--language", default="hu")
    args = p.parse_args()

    source = Path("data/extracted") / args.corpus / "languages" / f"{args.language}.jsonl"
    docs = [json.loads(l) for l in source.open(encoding="utf-8")]
    where = assign(docs)

    out = Path("data/processed") / f"{args.corpus}_{args.language}" / "target"
    (out / "splits").mkdir(parents=True, exist_ok=True)
    manifest = {"settings": {"budget": BUDGET, "dev_ratio": DEV_RATIO, "seed": SEED}, "splits": {}}
    for split in ("train", "dev", "test"):
        part = sorted((d for d in docs if where[d["doc"]] == split), key=lambda d: -d["words"])
        with (out / "splits" / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for d in part:
                f.write(json.dumps({**d, "split": split}, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {
            "documents": len(part),
            "words": sum(d["words"] for d in part),
            "docs": [d["doc"] for d in part],
        }
        print(f"  {split:5} {len(part):3} docs  {manifest['splits'][split]['words']:9,} words")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
