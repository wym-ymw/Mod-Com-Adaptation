import json

from safetensors.torch import load_file


def load(adapter_dir):
    # one adapter's config and tensors
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    return config, load_file(adapter_dir / "adapter_model.safetensors")


def merge(model, adapter_dirs):
    # add (alpha/r) * B @ A of every adapter into the model's weights, in place
    for adapter_dir in adapter_dirs:
        config, tensors = load(adapter_dir)
        scaling = config["lora_alpha"] / config["r"]
        for key, a in tensors.items():
            if ".lora_A." not in key:
                continue
            b = tensors[key.replace(".lora_A.", ".lora_B.")]
            path = key.split("base_model.model.")[1].split(".lora_A.")[0]
            target = model.get_submodule(path)
            delta = (b.float() @ a.float()) * scaling
            target.weight.data += delta.to(target.weight.dtype).to(target.weight.device)
