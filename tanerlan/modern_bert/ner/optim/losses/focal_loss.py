from typing import Literal, cast, override

import torch
import torch.nn.functional as F
from torch import nn
from transformers.utils.generic import maybe_autocast


class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss based on Lin et al. (2017).

    Focal loss addresses class imbalance by dynamically scaling down the cross entropy loss
    for well-classified examples, focusing training on hard negatives.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        """
        Initializes the FocalLoss module.

        Args:
            weight (torch.Tensor | None): Manual rescaling weight for each class.
                Must be a 1D tensor of size `num_classes`. Defaults to None.
            gamma (float): Focusing parameter for modulating factor (1 - p_t). Defaults to 2.0.
            ignore_index (int): Specifies a target value that is ignored and does not
                contribute to the input gradient. Defaults to -100.
            reduction (Literal["none", "mean", "sum"]): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. Defaults to 'mean'.
            reweight (bool): If True and reduction is 'mean', normalizes the loss
                by the sum of the weights of the target elements. Defaults to False.
        """
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        if weight is not None:
            if weight.ndim != 1:
                raise ValueError(f"`weight` must be a 1D tensor, got {weight.shape}")
            self.register_buffer("weight", weight.clone().detach(), persistent=False)
        return

    @override
    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the Focal Loss given raw logits and targets.

        Args:
            logits (torch.Tensor): Predicted unnormalized scores.
                Expected shape: (B, C, d1, d2, ...) or (B, C) where C is `num_classes`.
            target (torch.Tensor): Ground truth class indices.
                Expected shape: (B, d1, d2, ...) or (B,). Must have one fewer dimension than logits.

        Returns:
            torch.Tensor: The computed focal loss, reduced as specified by the `reduction` parameter.
        """
        if logits.ndim < 2:
            raise ValueError(
                f"`logits` must have shape (B, C, ...) or (B, C), got {logits.shape}"
            )
        if target.ndim != logits.ndim - 1:
            raise ValueError(
                f"`logits` must be {target.ndim + 1}D tensor, got {target.ndim}D"
            )

        if logits.ndim > 2:
            B, num_classes, *_ = logits.shape
            logits = (
                logits.view(B, num_classes, -1)
                .permute(0, 2, 1)
                .reshape(-1, num_classes)
            )  # [total_elements, num_classes]

        target = target.view(-1)  # [total_elements]

        if target.size(0) != logits.size(0):
            raise ValueError(
                "After flattening, `target` and `logits` must have "
                f"the same number of elements, got {target.size(0)} and {logits.size(0)}"
            )

        valid_mask = target != self.ignore_index

        if not valid_mask.any():
            raise ValueError(
                "All targets are ignored (equal to `ignore_index`), cannot compute loss"
            )

        logits = logits[valid_mask]
        target = target[valid_mask]

        logits_dtype = logits.dtype

        device_type = (
            logits.device.type
            if isinstance(logits.device.type, str) and logits.device.type != "mps"
            else "cpu"
        )
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            logits = logits.float()

            probs = F.softmax(logits, dim=1)
            log_probs = F.log_softmax(logits, dim=1)

            ce_loss = -log_probs[
                torch.arange(target.size(0), device=target.device), target
            ]
            target_probs = probs[
                torch.arange(target.size(0), device=target.device), target
            ]

            focal_factor = (1.0 - target_probs).pow(self.gamma)

            loss = focal_factor * ce_loss

            # if self.weight is not None:
            if hasattr(self, "weight") and self.weight is not None:
                weight = cast(torch.Tensor, self.weight)
                alpha_t = weight[target]
                loss = alpha_t * loss
            else:
                alpha_t = None

            match self.reduction:
                case "none":
                    reduced_loss = loss
                case "sum":
                    reduced_loss = loss.sum()
                case "mean":
                    if alpha_t is not None:
                        reduced_loss = loss.sum() / alpha_t.sum().clamp_min(1e-8)
                    else:
                        reduced_loss = loss.mean()
                case _:
                    raise ValueError(f"Unsupported reduction: {self.reduction}")
        return reduced_loss.to(logits_dtype)
