import argparse
import json
import urllib.request
from pathlib import Path

import yaml

CORPORA = {"dgt": ("DGT", "v2021"), "books": ("Books", "v1")}
STATS = "https://raw.githubusercontent.com/Helsinki-NLP/OPUS/main/corpus/{}/{}/statistics.yaml"
OUT = Path("data/raw")


def catalogue(corpus, version):
    # every monolingual language entry of an OPUS release
    raw = urllib.request.urlopen(STATS.format(corpus, version), timeout=300)
    stats = yaml.load(raw.read().decode(), Loader=yaml.BaseLoader)
    return stats["monolingual"]


def archive(entry, path):
    # download one language's zip unless it is already here
    url = entry["downloads"]["xml"]["url"]
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return url, int(entry["downloads"]["xml"]["size"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", nargs="+", default=["books", "dgt"])
    args = p.parse_args()

    for corpus in args.corpus:
        name, version = CORPORA[corpus]
        entries = catalogue(name, version)
        out = OUT / corpus
        (out / "archives").mkdir(parents=True, exist_ok=True)

        print(f"\n{name} {version}: {len(entries)} languages")
        index = {}
        for lang in sorted(entries):
            url, size = archive(entries[lang], out / "archives" / f"{lang}.zip")
            index[lang] = {
                "url": url,
                "size": size,
                "opus_files": int(entries[lang]["files"]),
                "opus_sentences": int(entries[lang]["sentences"]),
            }
            print(f"  {lang:4} {size / 1048576:8.1f} MB", flush=True)
        (out / "index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
