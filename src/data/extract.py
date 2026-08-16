# reference: https://github.com/Helsinki-NLP/OpusTools/blob/master/opustools_pkg/opustools/opus_cat.py
import argparse
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

RAW = Path("data/raw")
OUT = Path("data/extracted")


def sentence(node):
    # one <s> as whitespace-normalised text
    return " ".join("".join(node.itertext()).split())


def paragraphs(node):
    # sentences grouped as the XML groups them, one list per <p>
    out = []
    for child in node:
        if child.tag == "s":
            text = sentence(child)
            if text:
                out.append([text])
        elif child.tag == "p":
            texts = [t for t in map(sentence, child.iter("s")) if t]
            if texts:
                out.append(texts)
        else:
            out.extend(paragraphs(child))
    return out


def strip_header(paras):
    # drop the source, title and author lines Books puts on top
    if paras and paras[0][0].startswith("Source:"):
        return paras[3:], sum(len(p) for p in paras[:3])
    return paras, 0


def convert(corpus, lang, zip_path, out_path):
    # parse one language's zip into one JSON Lines file
    stats = {"documents": 0, "sentences": 0, "words": 0, "chars": 0, "header_lines": 0, "empty": 0, "unparsed": []}

    with zipfile.ZipFile(zip_path) as z, out_path.open("w", encoding="utf-8") as f:
        for member in sorted(m for m in z.namelist() if m.endswith(".xml")):
            name = Path(member).stem
            try:
                paras = paragraphs(ET.fromstring(z.read(member)))
            except ET.ParseError:
                stats["unparsed"].append(name)
                continue
            if corpus == "books":
                paras, dropped = strip_header(paras)
                stats["header_lines"] += dropped
            if not paras:
                stats["empty"] += 1
                continue

            text = "\n".join(" ".join(p) for p in paras)
            record = {
                "doc": name,
                "lang": lang,
                "text": text,
                "sentences": sum(len(p) for p in paras),
                "words": len(text.split()),
                "chars": len(text),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["documents"] += 1
            for key in ("sentences", "words", "chars"):
                stats[key] += record[key]
    return stats


def check(stats, entry):
    # compare the parse against OPUS's own document and sentence counts
    docs = stats["documents"] + stats["empty"] + len(stats["unparsed"])
    sents = stats["sentences"] + stats["header_lines"]
    if docs != entry["opus_files"] or sents != entry["opus_sentences"]:
        return f"MISMATCH {docs} docs vs {entry['opus_files']}, {sents} sent vs {entry['opus_sentences']}"
    return "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", nargs="+", default=["books", "dgt"])
    args = p.parse_args()

    for corpus in args.corpus:
        index = json.loads((RAW / corpus / "index.json").read_text())
        out = OUT / corpus
        (out / "languages").mkdir(parents=True, exist_ok=True)

        print(f"\n{corpus}: {len(index)} languages")
        manifest = {}
        for lang in sorted(index):
            stats = convert(corpus, lang, RAW / corpus / "archives" / f"{lang}.zip", out / "languages" / f"{lang}.jsonl")
            note = check(stats, index[lang])
            manifest[lang] = {**stats, "check": note}
            print(f"  {lang:5}{stats['documents']:8,} docs{stats['sentences']:12,} sentences{stats['words']:14,} words   {note}", flush=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()


# Books/raw/en/Poe_Edgar_Allan-Fall_of_the_House_of_Usher.xml, cut short:
#
#   <text>
#    <head>
#     <meta> The Fall of the House of Usher
#    by Edgar Allan Poe
#    Aligned by: Andras Farkas (fully reviewed)
#    </meta>
#    </head>
#    <body>
#     <s id="s1">Source: Project GutenbergAudiobook available here</s>
#     <s id="s2">The Fall of the House of Usher</s>
#     <s id="s3">Edgar Allan Poe</s>
#     <s id="s4">Son coeur est un luth suspendu; Sitot qu'on le touche il resonne.</s>
#     <p id="p22">
#      <s id="s22.0">I looked upon the scene before me.</s>
#      <s id="s22.1">I know not how it was.</s>
#     </p>
#    </body>
#   </text>
