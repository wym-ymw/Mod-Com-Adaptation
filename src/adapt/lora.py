import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.compose.merge import merge

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CACHE = Path("data/packed")
SEQ_LEN = 1024
LEARNING_RATE = 1e-4
WARMUP = 0.03
GRAD_CLIP = 1.0
DROPOUT = 0.05
EVAL_EVERY = 25
DEV_BATCHES = 512


def pack(path, tok, cache_dir):
    # tokenize a corpus, join documents with EOS, cut into equal-length sequences
    # the whole path is the cache key: corpora share file names like splits/train.jsonl
    cache = cache_dir / f"{str(path).replace('/', '-')}-{SEQ_LEN}.npy"
    if cache.exists():
        return np.load(cache)

    stream = []
    texts = [json.loads(line)["text"] for line in path.open(encoding="utf-8")]
    for start in range(0, len(texts), 256):
        for ids in tok(texts[start : start + 256], add_special_tokens=False).input_ids:
            stream.extend(ids)
            stream.append(tok.eos_token_id)

    n = len(stream) // SEQ_LEN
    packed = np.asarray(stream[: n * SEQ_LEN], dtype=np.int32).reshape(n, SEQ_LEN)
    cache.parent.mkdir(parents=True, exist_ok=True)
    scratch = cache.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(scratch, packed)
    scratch.replace(cache)
    print(f"  {path}: {len(texts):,} documents, {len(stream):,} tokens, {n:,} sequences")
    return packed


def batches(packs, size, device):
    # fixed-size batches of packed sequences, on the device
    for start in range(0, len(packs) - size + 1, size):
        yield torch.from_numpy(packs[start : start + size].astype(np.int64)).to(device)


@torch.no_grad()
def evaluate(model, dev, size, device):
    # mean token NLL on the held-out sequences, in nats
    model.eval()
    total, count = 0.0, 0
    for ids in batches(dev, size, device):
        logits = model(input_ids=ids).logits[:, :-1]
        total += torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), ids[:, 1:].reshape(-1), reduction="sum").item()
        count += ids[:, 1:].numel()
    model.train()
    return total / count


def learning_rate(step, total):
    # linear warmup over the first WARMUP of the run, then linear decay to zero
    warmup = max(1, int(total * WARMUP))
    if step < warmup:
        return LEARNING_RATE * (step + 1) / warmup
    return LEARNING_RATE * max(0.0, (total - step) / (total - warmup))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--model", default="utter-project/EuroLLM-1.7B")
    p.add_argument("--merge", nargs="*", default=[])
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulate", type=int, default=4)
    p.add_argument("--checkpointing", action="store_true")
    p.add_argument("--adapter-root", type=Path, default=Path("outputs/adapters"))
    args = p.parse_args()

    root = args.adapter_root / args.model.replace("/", "_") / "lora"

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.model_max_length = int(1e12)

    print(f"packing ({args.model})")
    cache = CACHE / args.model.replace("/", "_")
    train, dev = pack(args.train, tok, cache), pack(args.dev, tok, cache)
    dev = dev[:: max(1, len(dev) // DEV_BATCHES)][:DEV_BATCHES]
    print(f"  train {len(train):,} sequences, dev {len(dev):,} sequences")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    if args.checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if args.merge:
        merge(model, [root / n for n in args.merge])
        print(f"  merged {args.merge}")

    # alpha tied to the rank keeps alpha/r = 2, so adapters of different rank stay addable
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=DROPOUT,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM")).to(device)
    trainable = [q for q in model.parameters() if q.requires_grad]
    for q in trainable:
        # fp32 master copies: bf16 would swallow the small updates
        q.data = q.data.float()
    total = sum(q.numel() for q in model.parameters())
    print(f"  {sum(q.numel() for q in trainable):,} trainable of {total:,}")

    per_update = args.batch_size * args.accumulate
    updates = int(len(train) * args.epochs) // per_update
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.0)

    out = root / args.name
    out.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    log = {"settings": settings, "updates": updates, "history": []}

    rng = np.random.default_rng(args.seed)
    order, cursor = rng.permutation(len(train)), 0
    best, started = float("inf"), time.time()
    model.train()

    for step in range(updates):
        lr = learning_rate(step, updates)
        for group in optimizer.param_groups:
            group["lr"] = lr

        for _ in range(args.accumulate):
            if cursor + args.batch_size > len(order):
                order, cursor = rng.permutation(len(train)), 0
            ids = torch.from_numpy(train[order[cursor : cursor + args.batch_size]].astype(np.int64)).to(device)
            cursor += args.batch_size
            (model(input_ids=ids, labels=ids).loss / args.accumulate).backward()

        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if (step + 1) % EVAL_EVERY == 0 or step + 1 == updates:
            nll = evaluate(model, dev, args.batch_size, device)
            log["history"].append({"step": step + 1, "dev_nll": nll, "lr": lr})
            keep = nll < best
            if keep:
                best = nll
                model.save_pretrained(out)
            log["best_dev_nll"] = best
            (out / "training.json").write_text(json.dumps(log, indent=2))
            print(f"  step {step + 1:5}/{updates}  dev nll {nll:.4f}  ppl {math.exp(nll):8.2f}  {time.time() - started:6.0f}s{'  *' if keep else ''}", flush=True)

    print(f"\nbest dev nll {best:.4f}, adapter in {out}")


if __name__ == "__main__":
    main()
