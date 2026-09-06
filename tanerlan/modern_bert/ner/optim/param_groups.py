import re
from typing import Any

from torch import nn

_NO_DECAY_KEYWORDS = {"bias", "ln", "norm", "emb"}
_LAYER_RE = re.compile(r"^layers\.(\d+)\.")


def _backbone_depth(relative_name: str, num_layers: int) -> int:
    """Глубина параметра тушки: embeddings=0, layers.i=i+1, остальное (final_norm)=L+1."""
    if relative_name.startswith("embeddings."):
        return 0
    match = _LAYER_RE.match(relative_name)
    if match is not None:
        return int(match.group(1)) + 1
    return num_layers + 1


def _group_name(is_head: bool, depth: int | None, num_layers: int, wd: float) -> str:
    decay_suffix = "decay" if wd > 0 else "no_decay"
    if is_head:
        return f"head/{decay_suffix}"
    if depth is None:
        return f"backbone/{decay_suffix}"
    if depth == 0:
        block = "embeddings"
    elif depth == num_layers + 1:
        block = "final_norm"
    else:
        block = f"layer_{depth - 1:02d}"
    return f"backbone/{block}/{decay_suffix}"


def create_param_groups(
    model: nn.Module,
    weight_decay: float,
    backbone_lr: float,
    head_lr: float,
    layer_lr_decay: float | None = None,
    no_decay_keywords: set[str] | None = None,
) -> list[dict[str, Any]]:
    no_decay = _NO_DECAY_KEYWORDS | (no_decay_keywords or set())
    backbone_prefix = f"{getattr(model, 'base_model_prefix', 'model')}."
    use_llrd = layer_lr_decay is not None and layer_lr_decay < 1.0
    num_layers = int(getattr(getattr(model, "config", None), "num_hidden_layers", 0))
    if use_llrd and num_layers <= 0:
        raise ValueError("layer_lr_decay requires model.config.num_hidden_layers > 0")

    grouped: dict[tuple[bool, int | None, float], dict[str, Any]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = not name.startswith(backbone_prefix)
        wd = 0.0 if any(keyword in name for keyword in no_decay) else weight_decay

        depth: int | None = None
        scale = 1.0
        if not is_head and use_llrd:
            depth = _backbone_depth(name[len(backbone_prefix) :], num_layers)
            # верхний слой (depth=L) и final_norm (depth=L+1) -> 1, слой i (0-based) -> decay ** (L-1-i),
            # эмбеддинги -> decay ** L
            scale = float(layer_lr_decay) ** max(num_layers - depth, 0)
        key = (is_head, depth, wd)
        if key not in grouped:
            group: dict[str, Any] = {
                "params": [],
                "lr": head_lr if is_head else backbone_lr * scale,
                "weight_decay": wd,
                "name": _group_name(is_head, depth, num_layers, wd),
            }
            if is_head:
                group["is_head"] = True
            elif use_llrd:
                group["lr_scale"] = scale
            grouped[key] = group
        grouped[key]["params"].append(param)
    return list(grouped.values())
