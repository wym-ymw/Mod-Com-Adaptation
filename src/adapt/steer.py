import argparse
import json
import random
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = Path("data/extracted")
OUT = Path("outputs/adapters")
SEQ_LEN = 1024
MAX_WORDS = 1_000_000
SEED = 0


def slices(spec, max_words):
    # documents named by a corpus:lang,lang spec, sampled to a word budget each
    corpus, _, langs = spec.partition(":")
    names = langs.split(",")
    per = max(1, max_words // len(names))
    chosen = []
    for lang in names:
        docs = [json.loads(l) for l in (DATA / corpus / "languages" / f"{lang}.jsonl").open(encoding="utf-8")]
        docs.sort(key=lambda d: d["doc"])
        random.Random(SEED).shuffle(docs)
        words = 0
        for d in docs:
            chosen.append(d)
            words += d["words"]
            if words >= per:
                break
    return chosen


@torch.no_grad()
def mean_hidden(model, tok, docs, batch_size, device):
    # mean hidden state per layer over every token of a corpus slice
    stream = []
    for d in docs:
        stream.extend(tok(d["text"], add_special_tokens=False).input_ids)
        stream.append(tok.eos_token_id)
    n = len(stream) // SEQ_LEN
    blocks = torch.tensor(stream[: n * SEQ_LEN]).view(n, SEQ_LEN)

    total, count = None, 0
    for start in tqdm(range(0, n, batch_size), desc="  blocks", unit="batch", ascii=True, ncols=76, mininterval=30):
        ids = blocks[start : start + batch_size].to(device)
        hidden = model(input_ids=ids, output_hidden_states=True).hidden_states
        summed = torch.stack([h.float().sum(dim=(0, 1)) for h in hidden])
        total = summed if total is None else total + summed
        count += ids.numel()
    return total / count, count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--positive", required=True)
    p.add_argument("--negative", required=True)
    p.add_argument("--model", default="utter-project/EuroLLM-1.7B")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--adapter-root", type=Path, default=OUT)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.model_max_length = int(1e12)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.to(device).eval()

    sides = {}
    for side, spec in (("positive", args.positive), ("negative", args.negative)):
        docs = slices(spec, MAX_WORDS)
        print(f"{side}: {spec}  {len(docs)} documents, {sum(d['words'] for d in docs):,} words")
        sides[side], tokens = mean_hidden(model, tok, docs, args.batch_size, device)
        print(f"  {tokens:,} tokens")

    vector = sides["positive"] - sides["negative"]
    out = args.adapter_root / args.model.replace("/", "_") / "steer" / args.name
    out.mkdir(parents=True, exist_ok=True)
    save_file({"steering": vector.cpu()}, out / "steering.safetensors")
    # norm relative to the states it will be added to: weight 1 only makes sense if this is small
    scale = (vector.norm(dim=1) / sides["negative"].norm(dim=1)).tolist()
    (out / "steering.json").write_text(json.dumps({
        "kind": "steering", "model": args.model,
        "positive": args.positive, "negative": args.negative,
        "layers": vector.shape[0], "width": vector.shape[1],
        "max_words": MAX_WORDS, "seed": SEED,
        "relative_norm": [round(s, 4) for s in scale],
    }, indent=2))
    print(f"\n{vector.shape[0]} layers x {vector.shape[1]}  ->  {out}")
    print(f"  relative norm per layer: min {min(scale):.3f}  median {sorted(scale)[len(scale)//2]:.3f}  max {max(scale):.3f}")


if __name__ == "__main__":
    main()
