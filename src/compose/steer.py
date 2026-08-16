import json

from safetensors.torch import load_file


def load(adapter_dir):
    # one steering module's vectors and the record of what produced them
    meta = json.loads((adapter_dir / "steering.json").read_text())
    return meta, load_file(adapter_dir / "steering.safetensors")["steering"]


def steer(model, adapter_dirs):
    # add the sum of several steering vectors to each layer's output
    total = None
    for adapter_dir in adapter_dirs:
        _, vector = load(adapter_dir)
        total = vector if total is None else total + vector
    if total is None:
        return []

    handles = []
    for i, layer in enumerate(model.model.layers):
        # hidden_states index i + 1 is the output of layer i; index 0 is the embedding
        weight_like = next(layer.parameters())
        v = total[i + 1].to(dtype=weight_like.dtype, device=weight_like.device)

        def hook(module, args, output, v=v):
            if isinstance(output, tuple):
                return (output[0] + v,) + output[1:]
            return output + v

        handles.append(layer.register_forward_hook(hook))
    return handles

