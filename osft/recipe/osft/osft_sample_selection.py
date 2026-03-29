# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""On-policy SFT sample selection: positive (correct) rollouts and optional negative rollouts."""

from __future__ import annotations

import torch

from verl import DataProto

# Per-sequence sign s: loss uses (-log_prob * s) aggregated over tokens, so s=+1 is standard CE
# on the completion; s=-w minimizes w * log_prob (discourages the completion).
OSFT_LOSS_SIGN_KEY = "osft_loss_sign"


def apply_osft_sample_selection(
    batch: DataProto,
    *,
    enable_negative_sample_training: bool,
    negative_sample_loss_scale: float,
    dp_world_size: int,
    rollout_n: int | None,
) -> DataProto:
    """
    Filter rollouts for OSFT training (same positive rule as legacy `_select_best_of_n`).

    Positive rule per uid (same prompt):
      - C = count with sequence score > 0; N = rollout count
      - C == 0: drop entire uid
      - C == N: skip entire uid (all-correct problems are not trained)
      - 0 < C < N: keep all correct samples

    If ``enable_negative_sample_training`` is True, also keep every incorrect rollout (score <= 0)
    for uids that have at least one kept positive sample. Each row gets ``OSFT_LOSS_SIGN_KEY``:
    +1.0 for positives, ``-negative_sample_loss_scale`` for negatives.

    When ``enable_negative_sample_training`` is False, behavior matches the legacy path: only
    positive rows are kept and ``OSFT_LOSS_SIGN_KEY`` is not added.
    """
    uids = [str(u) for u in batch.non_tensor_batch["uid"]]
    scores = batch.batch["token_level_scores"].sum(-1)
    resp_len = batch.batch["response_mask"].sum(-1)

    per_uid: dict[str, list[tuple[int, float, int]]] = {}
    for i, uid in enumerate(uids):
        s = float(scores[i].item())
        l = int(resp_len[i].item())
        per_uid.setdefault(uid, []).append((i, s, l))

    positive_indices: list[int] = []
    for uid, lst in per_uid.items():
        correct = [(idx, s, l) for (idx, s, l) in lst if s > 0]
        c = len(correct)
        if c == 0:
            continue
        n = rollout_n if rollout_n is not None else len(lst)
        if c == n:
            continue
        positive_indices.extend(idx for (idx, _, _) in correct)

    if not positive_indices:
        return batch.select_idxs([])

    positive_set = set(positive_indices)
    merged_indices: list[int]

    if enable_negative_sample_training:
        negative_indices: list[int] = []
        for uid, lst in per_uid.items():
            if not any(idx in positive_set for (idx, _, _) in lst):
                continue
            for idx, s, _ in lst:
                if s <= 0:
                    negative_indices.append(idx)
        merged_indices = sorted(positive_set | set(negative_indices))
    else:
        merged_indices = sorted(positive_set)

    if dp_world_size > 1:
        trim_len = (len(merged_indices) // dp_world_size) * dp_world_size
        if trim_len == 0:
            return batch.select_idxs([])
        merged_indices = merged_indices[:trim_len]

    out = batch.select_idxs(merged_indices)

    if enable_negative_sample_training:
        device = out.batch["token_level_scores"].device
        signs: list[float] = []
        for idx in merged_indices:
            seq_score = float(scores[idx].item())
            if seq_score > 0:
                signs.append(1.0)
            else:
                signs.append(-float(negative_sample_loss_scale))
        if any(s < 0 for s in signs):
            out.batch[OSFT_LOSS_SIGN_KEY] = torch.tensor(signs, device=device, dtype=torch.float32)

    return out
