# -*- coding: utf-8 -*-
"""
Training and encoding loops for temporal NeuroBridge models.

The functions in this module intentionally remain lightweight. They do not
hide the scientific choices made by each experiment; instead, they provide
the repeated mechanics:

- move a batch to the selected device;
- run one training epoch;
- encode every temporal window;
- collect aligned metadata.

Supported loss calling conventions
----------------------------------
1. Label-based objective:
       loss_fn(embedding, batch["label"])

2. Similarity-based objective:
       similarity = similarity_builder(batch_on_device)
       loss_fn(embedding, similarity)

3. Batch-aware objective:
       loss_fn(embedding, batch_on_device)

Only one custom calling convention may be selected at a time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


Batch = Mapping[str, Any]


def move_batch_to_device(
    batch: Batch,
    device: torch.device | str,
) -> dict[str, Any]:
    """
    Move tensor-like batch values to a device.

    Non-tensor metadata is preserved unchanged.
    """
    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping.")

    moved: dict[str, Any] = {}

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value

    return moved


def _validate_training_inputs(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    *,
    similarity_builder: Callable | None,
    batch_loss: bool,
) -> None:
    """Validate the public training-loop arguments."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not hasattr(dataloader, "__iter__"):
        raise TypeError("dataloader must be iterable.")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer.")
    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable.")
    if similarity_builder is not None and not callable(similarity_builder):
        raise TypeError("similarity_builder must be callable.")
    if batch_loss and similarity_builder is not None:
        raise ValueError(
            "batch_loss=True and similarity_builder are mutually exclusive."
        )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device | str = "cpu",
    similarity_builder: Callable[[dict[str, Any]], torch.Tensor] | None = None,
    batch_loss: bool = False,
    grad_clip_norm: float | None = None,
) -> float:
    """
    Train a temporal encoder for one epoch.

    Parameters
    ----------
    model:
        Temporal encoder returning one embedding per input window.

    dataloader:
        Iterable yielding dictionaries. Every batch must contain ``"x"``.

    optimizer:
        PyTorch optimizer.

    loss_fn:
        Objective function. Its arguments depend on ``batch_loss`` and
        ``similarity_builder``.

    device:
        CPU or CUDA device.

    similarity_builder:
        Optional callable receiving the batch already moved to ``device`` and
        returning a pairwise similarity matrix.

    batch_loss:
        When True, call ``loss_fn(embedding, batch_on_device)``.

    grad_clip_norm:
        Optional maximum global gradient norm.

    Returns
    -------
    float
        Mean batch loss for the epoch.
    """
    _validate_training_inputs(
        model,
        dataloader,
        optimizer,
        loss_fn,
        similarity_builder=similarity_builder,
        batch_loss=batch_loss,
    )

    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be strictly positive.")

    model.to(device)
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_index, batch in enumerate(dataloader):
        batch_on_device = move_batch_to_device(batch, device)

        if "x" not in batch_on_device:
            raise ValueError(
                f"Batch {batch_index} does not contain the required key 'x'."
            )

        x = batch_on_device["x"]

        if not isinstance(x, torch.Tensor):
            raise TypeError("batch['x'] must be a torch.Tensor.")

        optimizer.zero_grad(set_to_none=True)
        embedding = model(x)

        if batch_loss:
            loss = loss_fn(embedding, batch_on_device)

        elif similarity_builder is not None:
            similarity = similarity_builder(batch_on_device)

            if not isinstance(similarity, torch.Tensor):
                raise TypeError(
                    "similarity_builder must return a torch.Tensor."
                )

            loss = loss_fn(embedding, similarity)

        else:
            if "label" not in batch_on_device:
                raise ValueError(
                    "batch must contain 'label' when no similarity_builder "
                    "is provided and batch_loss is False."
                )

            loss = loss_fn(
                embedding,
                batch_on_device["label"],
            )

        if not isinstance(loss, torch.Tensor):
            raise TypeError("loss_fn must return a torch.Tensor.")
        if loss.ndim != 0:
            raise ValueError(
                f"loss_fn must return a scalar tensor, got shape {loss.shape}."
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss encountered in batch {batch_index}: {loss}."
            )

        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip_norm,
            )

        optimizer.step()

        batch_size = int(x.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("dataloader produced no samples.")

    return total_loss / total_samples


@torch.no_grad()
def encode_windows(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Encode all windows and collect metadata in matching order.

    Tensor metadata is concatenated along the batch dimension. Non-tensor
    metadata is collected in Python lists.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not hasattr(dataloader, "__iter__"):
        raise TypeError("dataloader must be iterable.")

    model.to(device)
    model.eval()

    embedding_batches: list[torch.Tensor] = []
    tensor_metadata: dict[str, list[torch.Tensor]] = {}
    object_metadata: dict[str, list[Any]] = {}

    for batch_index, batch in enumerate(dataloader):
        batch_on_device = move_batch_to_device(batch, device)

        if "x" not in batch_on_device:
            raise ValueError(
                f"Batch {batch_index} does not contain the required key 'x'."
            )

        x = batch_on_device["x"]

        if not isinstance(x, torch.Tensor):
            raise TypeError("batch['x'] must be a torch.Tensor.")

        embedding = model(x)

        if not isinstance(embedding, torch.Tensor):
            raise TypeError("model must return a torch.Tensor.")
        if embedding.ndim != 2:
            raise ValueError(
                "model output must have shape "
                "(batch_size, embedding_dim)."
            )

        embedding_batches.append(embedding.detach().cpu())

        for key, value in batch.items():
            if key == "x":
                continue

            if isinstance(value, torch.Tensor):
                tensor_metadata.setdefault(key, []).append(
                    value.detach().cpu()
                )
            else:
                object_metadata.setdefault(key, []).append(value)

    if not embedding_batches:
        raise ValueError("dataloader produced no samples.")

    embeddings = torch.cat(embedding_batches, dim=0)

    metadata: dict[str, Any] = {
        key: torch.cat(values, dim=0)
        for key, values in tensor_metadata.items()
    }
    metadata.update(object_metadata)

    return embeddings, metadata
