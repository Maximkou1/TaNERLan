from typing import Any

from torch import nn


def create_param_groups(
    model: nn.Module,
    weight_decay: float,
    backbone_lr: float,
    embeddings_lr: float,
    embedding_keywords: str | set[str] | None = None,
    no_lr_keywords: set[str] | None = None,
    no_decay_keywords: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Create optimizer parameter groups for a PyTorch model with fine-grained weight decay control.

    Args:
        model (nn.Module): The PyTorch model containing the parameters to optimize.
        weight_decay (float): The default weight decay value to apply to parameters that are
            not excluded.
        backbone_lr (float): The learning rate to assign to all parameter groups.
        no_lr_keywords (set[str] | None, optional): A set of string keywords. If a parameter's
            name contains any of these keywords, its learning rate is set to 0.0.
            Defaults to None, which uses an empty set.
        no_decay_keywords (set[str] | None, optional): A set of string keywords. If a parameter's
            name contains any of these keywords, its weight decay is set to 0.0.
            If keywords are provided, they will be added to the default set, otherwise the default set is used.
            Default set of keywords:
            {"bias", "emb", "ln"}.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a parameter group
            compatible with PyTorch optimizers (e.g., `torch.optim.AdamW`). Each group contains:
            - "params": The parameter tensor.
            - "lr": The learning rate.
            - "weight_decay": The specific weight decay value (0.0 or the provided default).

    """
    no_decay_keywords_ = {"bias", "ln", "norm", "emb"}
    if no_decay_keywords is not None:
        no_decay_keywords_ = no_decay_keywords_.union(no_decay_keywords)

    no_lr_keywords_ = set()
    if no_lr_keywords is not None:
        no_lr_keywords_ = no_lr_keywords_.union(no_lr_keywords)

    if isinstance(embedding_keywords, str):
        embedding_keywords = {embedding_keywords}
    embedding_keywords_ = embedding_keywords or set()

    param_groups = []
    for name, param in model.named_parameters():
        if param.requires_grad is False:
            continue

        is_embedding = any(keyword in name for keyword in embedding_keywords_)
        
        if is_embedding:
            lr = embeddings_lr
        else:
            lr = (
                backbone_lr
                if not any(keyword in name for keyword in no_lr_keywords_)
                else 0.0
            )
            
        wd = (
            weight_decay
            if not any(keyword in name for keyword in no_decay_keywords_)
            else 0.0
        )

        param_group = {"params": param, "lr": lr, "weight_decay": wd}

        # add flag to indicate that this parameter group is for embeddings
        if is_embedding:
            param_group["is_embedding"] = True

        param_groups.append(param_group)

    fused_param_groups = _fuse_groups(param_groups)
    return fused_param_groups


def _fuse_groups(param_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fused_groups_dict: dict[str, dict[str, Any]] = {}
    for group in param_groups:
        group_key = ""
        for key, value in group.items():
            if key != "params":
                group_key += f"_{key}:{value}"

        if group_key not in fused_groups_dict:
            fused_groups_dict[group_key] = {"params": []}
            for k, v in group.items():
                if k != "params":
                    fused_groups_dict[group_key][k] = v

        fused_groups_dict[group_key]["params"].append(group["params"])
    return list(fused_groups_dict.values())


create_params_groups = create_param_groups  # Alias for backward compatibility
