import argparse
import json
import math
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

DATA = Path("data/extracted")
OUT = Path("outputs/probe")
WINDOW = 1024
MAX_WORDS = 250_000
SEED = 42


def sample(path, max_words):
    # documents in seeded random order, taken until the word budget is reached
    docs = [json.loads(l) for l in path.open(encoding="utf-8")]
    order = sorted(docs, key=lambda d: d["doc"])
    random.Random(SEED).shuffle(order)
    chosen, words = [], 0
    for d in order:
        chosen.append(d)
        words += d["words"]
        if words >= max_words:
            break
    return chosen, words


def stream(docs, tok, eos):
    # documents tokenized and concatenated, each token charged the characters it covers
    ids, chars, byts = [], [], []
    for d in docs:
        text = d["text"]
        encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        cut = 0
        for token, (_, end) in zip(encoded.input_ids, encoded.offset_mapping):
            ids.append(token)
            chars.append(end - cut)
            byts.append(len(text[cut:end].encode()))
            cut = end
        ids.append(eos)
        chars.append(0)
        byts.append(0)
    return ids, chars, byts


@torch.no_grad()
def score(model, ids, batch_size, device):
    # NLL per token, from windows cut like training sequences, first token unscored
    nll = [None] * len(ids)
    spans = [(b, min(b + WINDOW, len(ids))) for b in range(0, len(ids), WINDOW)]
    spans = [(b, e) for b, e in spans if e - b >= 2]

    for start in tqdm(range(0, len(spans), batch_size), desc="  windows", unit="batch", ascii=True, ncols=76, mininterval=30):
        batch = spans[start : start + batch_size]
        width = max(e - b for b, e in batch)
        block = torch.zeros(len(batch), width, dtype=torch.long)
        for row, (b, e) in enumerate(batch):
            block[row, : e - b] = torch.tensor(ids[b:e])
        block = block.to(device)

        logits = model(input_ids=block).logits
        for row, (b, e) in enumerate(batch):
            # position t predicts t+1, so logits sit one place left of their target
            length = e - b
            loss = torch.nn.functional.cross_entropy(logits[row, : length - 1].float(), block[row, 1:length], reduction="none")
            for offset, value in enumerate(loss.tolist()):
                nll[b + 1 + offset] = value
    return nll


def metrics(nll, chars, byts, documents, words):
    # corpus-level rates: sums over the text, never a mean of per-window rates
    scored = [i for i, v in enumerate(nll) if v is not None]
    total = {
        "nats": sum(nll[i] for i in scored),
        "tokens": len(scored),
        "chars": sum(chars[i] for i in scored),
        "bytes": sum(byts[i] for i in scored),
        "documents": documents,
        "words": words,
    }
    return {
        **total,
        "bpc": total["nats"] / math.log(2) / total["chars"],
        "bpb": total["nats"] / math.log(2) / total["bytes"],
        "ppl_token": math.exp(total["nats"] / total["tokens"]),
        "fertility": total["tokens"] / words,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="utter-project/EuroLLM-1.7B")
    p.add_argument("--corpus", nargs="+", default=["dgt", "books"])
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.model_max_length = int(1e12)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    eos = tok.eos_token_id

    settings = {"model": args.model, "window": WINDOW, "max_words": MAX_WORDS, "seed": SEED}
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / (args.model.replace("/", "_") + ".json")

    # resume, but only onto results these exact settings produced
    report = {"settings": settings, "results": {}}
    if out_path.exists():
        cached = json.loads(out_path.read_text())
        if cached["settings"] == settings:
            report = cached

    for corpus in args.corpus:
        manifest = json.loads((DATA / corpus / "manifest.json").read_text())
        print(f"\n{corpus}  ({args.model})")
        print(f"  {'lang':5} {'bpc':>6} {'bpb':>6} {'ppl_token':>9} {'fert':>5} {'tokens':>9} {'words':>9} {'docs':>7}")
        for lang in sorted(manifest):
            key = f"{corpus}/{lang}"
            if key not in report["results"]:
                docs, words = sample(DATA / corpus / "languages" / f"{lang}.jsonl", MAX_WORDS)
                ids, chars, byts = stream(docs, tok, eos)
                nll = score(model, ids, args.batch_size, device)
                report["results"][key] = metrics(nll, chars, byts, len(docs), words)
                out_path.write_text(json.dumps(report, indent=2))
            r = report["results"][key]
            print(f"  {lang:5} {r['bpc']:6.3f} {r['bpb']:6.3f} {r['ppl_token']:9.2f} {r['fertility']:5.2f} {r['tokens']:9,} {r['words']:9,} {r['documents']:7,}", flush=True)


if __name__ == "__main__":
    main()
