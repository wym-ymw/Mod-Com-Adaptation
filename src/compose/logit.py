import torch


def logit(base, experts):
    # log p = log p_base + sum_i (log p_i - log p_base), renormalised.
    # every term is normalised first, so an expert contributes its change in
    # log-probability and not the arbitrary offset a logit vector carries
    out = torch.log_softmax(base.float(), dim=-1)
    total = out
    for expert in experts:
        total = total + torch.log_softmax(expert.float(), dim=-1) - out
    return torch.log_softmax(total, dim=-1)
