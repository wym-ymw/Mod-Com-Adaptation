# Modular and Compositional Adaptation of a Multilingual LM

Adapt a pretrained multilingual LM independently for two factors, a language
and a genre, then compose the two modules on the small intersection domain and
measure how well the result models it.

The write-up is in [report.pdf](report.pdf).

## Structure

```
src/data/      download.py        OPUS raw archives, one zip per language
               extract.py         zips -> one JSONL per language, one line per document
               split_target.py    target domain -> train / dev / test
               split_modules.py   the two module corpora, word budgets matched
src/adapt/     lora.py            train one LoRA adapter on one corpus
               steer.py           extract a steering vector (comparison adapter)
src/compose/   merge.py           fold adapter updates into the backbone weights
               steer.py           add steering vectors to the residual stream
               logit.py           combine output distributions (comparison composer)
src/eval/      probe.py           score untuned backbones on every language of both corpora
               score.py           score one configuration on the target test split
```

`data/`, `logs/` and `outputs/` are kept empty here; the commands below fill them.

## Reproduce

Environment: `pip install -r requirements.txt`.
One 48GB GPU suffices. EuroLLM runs at the defaults, `--batch-size 8
--accumulate 4`. Qwen has a larger vocabulary and so needs more room for the
logits, `--checkpointing --batch-size 4 --accumulate 8`. The effective batch is
32 sequences either way.

Data: download OPUS, extract, build the target and module corpora.

```bash
python -m src.data.download
python -m src.data.extract
python -m src.data.split_target  --corpus books --language hu
python -m src.data.split_modules --corpus books --language hu
```

Backbone selection, one run per candidate model.

```bash
python -m src.eval.probe --model utter-project/EuroLLM-1.7B
```

The study on one backbone, for one seed. `--model` defaults to EuroLLM-1.7B;
pass `--model Qwen/Qwen3-4B-Base` to every command below for the second
backbone, and repeat the block for seeds 0, 1 and 2.

```bash
D=data/processed/books_hu

# the two factor modules, each on its own corpus
python -m src.adapt.lora --name books_adapter_r16_s0 --epochs 1 --seed 0 \
    --train $D/books_adapter/splits/train.jsonl --dev $D/books_adapter/splits/dev.jsonl
python -m src.adapt.lora --name hu_adapter_r16_s0 --epochs 1 --seed 0 \
    --train $D/hu_adapter/splits/train.jsonl --dev $D/hu_adapter/splits/dev.jsonl

# the two supervised runs: same data and budget, different starting point
python -m src.adapt.lora --name target_adapter_r16_s0 --epochs 5 --seed 0 \
    --train $D/target/splits/train.jsonl --dev $D/target/splits/dev.jsonl
python -m src.adapt.lora --name composed_target_r16_s0 --epochs 5 --seed 0 \
    --merge books_adapter_r16_s0 hu_adapter_r16_s0 \
    --train $D/target/splits/train.jsonl --dev $D/target/splits/dev.jsonl

# one score per adapted model, on the held-out target split
python -m src.eval.score                                             # base
python -m src.eval.score hu_adapter_r16_s0                           # language adapter
python -m src.eval.score books_adapter_r16_s0                        # genre adapter
python -m src.eval.score target_adapter_r16_s0                       # direct adapter
python -m src.eval.score --tag composed_r16_s0 \
    books_adapter_r16_s0 hu_adapter_r16_s0                           # composed, zero shot
python -m src.eval.score --tag composed_target_r16_s0 \
    books_adapter_r16_s0 hu_adapter_r16_s0 composed_target_r16_s0    # composed, few shot
```

The two comparison points that stay out of the main results.

```bash
python -m src.eval.score --compose logit \
    hu_adapter_r16_s0 books_adapter_r16_s0                           # comparison composer
python -m src.eval.score --kind steer hu books                       # comparison adapter
```

The second target language repeats all of the above with `--language fr`,
against a French language module trained on the same budget.

```bash
python -m src.data.split_target  --corpus books --language fr
python -m src.data.split_modules --corpus books --language fr
```

Outputs land under `outputs/` as
`adapters/<model>/{lora,steer}/<name>/` and
`eval/<model>/<kind>/<compose>/<tag>.json`; each eval JSON carries the
settings that produced it, per-book scores, and per-block records.
