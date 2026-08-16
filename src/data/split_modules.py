import argparse
import json
import random
from collections import Counter
from pathlib import Path

EXTRACTED = Path("data/extracted")
PROCESSED = Path("data/processed")
CORPORA = ("books", "dgt")
DEV_RATIO = 0.1
DEV_MIN_DOCS = 2
SEED = 42


def load(path):
    # every document of one language file
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def assign(docs, budget):
    # seeded random order fills train, then dev, until the budget is spent
    order = sorted(docs, key=lambda d: (d["lang"], d["doc"]))
    random.Random(SEED).shuffle(order)
    available = Counter(d["lang"] for d in docs)
    total, where = Counter(), {}
    for d in order:
        if total["train"] + total["dev"] + d["words"] > budget:
            continue
        if total["train"] + d["words"] <= budget * (1 - DEV_RATIO):
            split = "train"
        elif available[d["lang"]] >= DEV_MIN_DOCS:
            split = "dev"
        else:
            # a language with a single document stays in train, never dev-only
            split = "train"
        where[(d["lang"], d["doc"])] = split
        total[split] += d["words"]
    return where


def write(docs, where, out):
    # write one module's train and dev splits and its manifest
    (out / "splits").mkdir(parents=True, exist_ok=True)
    manifest = {"settings": {"seed": SEED, "dev_ratio": DEV_RATIO}, "splits": {}}
    for split in ("train", "dev"):
        part = [d for d in docs if where.get((d["lang"], d["doc"])) == split]
        with (out / "splits" / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for d in part:
                f.write(json.dumps({**d, "split": split}, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {
            "documents": len(part),
            "words": sum(d["words"] for d in part),
            "languages": sorted({d["lang"] for d in part}),
        }
        print(f"  {split:5} {len(part):6,} docs  {manifest['splits'][split]['words']:11,} words")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="books")
    p.add_argument("--language", default="hu")
    args = p.parse_args()
    other = next(c for c in CORPORA if c != args.corpus)

    # the genre module reads the target corpus without the target language
    corpus_docs = []
    for path in sorted((EXTRACTED / args.corpus / "languages").glob("*.jsonl")):
        if path.stem != args.language:
            corpus_docs += load(path)
    # the language module reads the target language of the other corpus
    language_docs = load(EXTRACTED / other / "languages" / f"{args.language}.jsonl")
    # both modules get the same word budget, so neither is stronger for having seen more
    budget = min(sum(d["words"] for d in corpus_docs), sum(d["words"] for d in language_docs))

    out = PROCESSED / f"{args.corpus}_{args.language}"
    for name, docs in ((f"{args.corpus}_adapter", corpus_docs),
                       (f"{args.language}_adapter", language_docs)):
        print(f"\n{name}: pool {len(docs):,} documents, {sum(d['words'] for d in docs):,} words")
        write(docs, assign(docs, budget), out / name)


if __name__ == "__main__":
    main()
