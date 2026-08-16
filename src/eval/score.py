import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from src.compose.logit import logit
from src.compose.merge import merge
from src.compose.steer import steer

DATA = Path("data/processed")
SEQ_LEN = 1024


def blocks(path, tok, seq_len):
    # cut each document into token blocks: same 1024 length, no overlap and nothing
    # prepended as in training, but never crossing a document, so a block belongs to one book
    out = []
    for line in path.open(encoding="utf-8"):
        doc = json.loads(line)
        text = doc["text"]
        encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded.input_ids, encoded.offset_mapping
        for start in range(0, len(ids), seq_len):
            end = min(start + seq_len, len(ids))
            if end - start < 2:
                continue
            stop = len(text) if end == len(ids) else offsets[end - 1][1]
            span = text[offsets[start][1] : stop]
            out.append({
                "doc": doc["doc"],
                "ids": ids[start:end],
                "scored": end - start - 1,
                "chars": len(span),
                "bytes": len(span.encode()),
            })
    return out


@torch.no_grad()
def run(models, items, batch_size, device):
    # NLL per block; models[0] is scored, any further models are logit-space experts
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    for start in tqdm(range(0, len(order), batch_size), desc="  blocks", unit="batch", ascii=True, ncols=76, mininterval=30):
        batch = [items[i] for i in order[start : start + batch_size]]
        width = max(len(item["ids"]) for item in batch)
        ids = torch.zeros(len(batch), width, dtype=torch.long)
        for row, item in enumerate(batch):
            ids[row, : len(item["ids"])] = torch.tensor(item["ids"])
        ids = ids.to(device)

        out = [model(input_ids=ids).logits for model in models]
        for row, item in enumerate(batch):
            # position t predicts t+1, so logits sit one place left of their target
            length, scored = len(item["ids"]), item["scored"]
            window = slice(length - scored - 1, length - 1)
            logprobs = logit(out[0][row, window], [o[row, window] for o in out[1:]])
            loss = torch.nn.functional.nll_loss(logprobs, ids[row, length - scored : length], reduction="sum")
            item["nats"] = loss.item()
            item["tokens"] = scored


def totals(items):
    # corpus-level rates: sums over the text, never a mean of per-block rates
    s = {k: sum(i[k] for i in items) for k in ("nats", "chars", "bytes", "tokens")}
    s["blocks"] = len(items)
    return {
        **s,
        "bpc": s["nats"] / math.log(2) / s["chars"],
        "bpb": s["nats"] / math.log(2) / s["bytes"],
        "ppl_token": math.exp(s["nats"] / s["tokens"]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("modules", nargs="*")
    p.add_argument("--kind", choices=("lora", "steer"), default="lora")
    p.add_argument("--compose", choices=("native", "logit"), default="native")
    p.add_argument("--model", default="utter-project/EuroLLM-1.7B")
    p.add_argument("--corpus", default="books")
    p.add_argument("--language", default="hu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--adapter-root", type=Path, default=Path("outputs/adapters"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/eval"))
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.model_max_length = int(1e12)
    source = args.adapter_root / args.model.replace("/", "_") / args.kind
    target = DATA / f"{args.corpus}_{args.language}" / "target" / "splits" / "test.jsonl"

    def backbone():
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        return model.to(device).eval()

    def attach(model, names):
        # put modules of this kind into one model, at full strength
        apply = steer if args.kind == "steer" else merge
        apply(model, [source / n for n in names])

    if args.compose == "native":
        # every module goes into one model, so the sum happens inside it
        model = backbone()
        attach(model, args.modules)
        models = [model]
    else:
        # one model per module, and only their output distributions are combined
        models = [backbone()]
        for n in args.modules:
            expert = backbone()
            attach(expert, [n])
            models.append(expert)
    print(f"{args.kind} {args.compose}: {args.modules or 'none'}")

    items = blocks(target, tok, SEQ_LEN)
    print(f"{target}: {len(items):,} blocks of at most {SEQ_LEN} tokens")
    run(models, items, args.batch_size, device)

    by_doc = {}
    for item in items:
        by_doc.setdefault(item["doc"], []).append(item)

    tag = args.tag or ("base" if not args.modules else "+".join(args.modules))
    report = {
        "settings": {"model": args.model, "kind": args.kind, "compose": args.compose,
                     "modules": args.modules, "corpus": args.corpus, "language": args.language},
        "overall": totals(items),
        "by_doc": {doc: totals(v) for doc, v in sorted(by_doc.items())},
        "blocks": [{k: i[k] for k in ("doc", "nats", "chars", "bytes", "tokens")} for i in items],
    }

    out = args.out_dir / args.model.replace("/", "_") / args.kind / args.compose
    out.mkdir(parents=True, exist_ok=True)
    out = out / f"{tag}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    o = report["overall"]
    print(f"\n{tag}   bpb {o['bpb']:.4f}   bpc {o['bpc']:.4f}   ppl_token {o['ppl_token']:.2f}")
    for doc, s in sorted(report["by_doc"].items(), key=lambda x: -x[1]["bpb"]):
        print(f"  {s['bpb']:.4f}  {s['blocks']:4} blocks  {doc[:46]}")


if __name__ == "__main__":
    main()
