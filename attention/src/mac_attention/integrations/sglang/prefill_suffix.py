from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch


@dataclass(frozen=True)
class PrefillSuffixSegment:
    batch_index: int
    source_start: int
    update_start: int
    length: int
    kv_source_start: int
    kv_len: int


@dataclass(frozen=True)
class PrefillSuffixPlan:
    segments: tuple[PrefillSuffixSegment, ...]
    total_len: int
    req_indices: tuple[int, ...]
    request_lengths: tuple[int, ...]
    lens: tuple[int, ...]
    offsets: tuple[int, ...]
    kv_offsets: tuple[int, ...]


def _cpu_int_list(value: Any, batch_size: int) -> Optional[list[int]]:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            return None
        return [int(x) for x in value[:batch_size].tolist()]
    try:
        return [int(x) for x in value[:batch_size]]
    except Exception:
        return None


def build_prefill_suffix_plan(
    forward_batch: Any, batch_size: int, total_q_len: int
) -> Optional[PrefillSuffixPlan]:
    """Return the compact prefill suffix cache-update plan.

    ``None`` means no scheduler gate is available and callers should preserve the
    legacy full-extend behavior. An empty plan means the scheduler explicitly
    skipped this chunk for all requests.
    """

    update_starts = _cpu_int_list(
        getattr(forward_batch, "mac_prefill_cache_update_starts", None),
        batch_size,
    )
    if update_starts is None:
        return None

    prefix_lens = _cpu_int_list(
        getattr(forward_batch, "extend_prefix_lens_cpu", None),
        batch_size,
    )
    extend_lens = _cpu_int_list(
        getattr(forward_batch, "extend_seq_lens_cpu", None),
        batch_size,
    )
    if prefix_lens is None or extend_lens is None:
        return None
    if (
        len(update_starts) < batch_size
        or len(prefix_lens) < batch_size
        or len(extend_lens) < batch_size
    ):
        return None

    extend_lens = [max(0, int(x)) for x in extend_lens[:batch_size]]
    if sum(extend_lens) != int(total_q_len):
        return None

    segments: list[PrefillSuffixSegment] = []
    offsets = [0]
    kv_offsets = [0]
    q_base = 0
    kv_base = 0
    for batch_index in range(batch_size):
        prefix_len = max(0, int(prefix_lens[batch_index]))
        extend_len = extend_lens[batch_index]
        kv_len = prefix_len + extend_len
        update_start = int(update_starts[batch_index])
        if update_start >= 0 and extend_len > 0:
            local_start = update_start - prefix_len
            if local_start < 0:
                local_start = 0
                update_start = prefix_len
            if local_start < extend_len:
                update_len = extend_len - local_start
                segments.append(
                    PrefillSuffixSegment(
                        batch_index=batch_index,
                        source_start=q_base + local_start,
                        update_start=update_start,
                        length=update_len,
                        kv_source_start=kv_base,
                        kv_len=kv_len,
                    )
                )
                offsets.append(offsets[-1] + update_len)
                kv_offsets.append(kv_offsets[-1] + kv_len)
        q_base += extend_len
        kv_base += kv_len

    return PrefillSuffixPlan(
        segments=tuple(segments),
        total_len=offsets[-1],
        req_indices=tuple(s.batch_index for s in segments),
        request_lengths=tuple(s.update_start for s in segments),
        lens=tuple(s.length for s in segments),
        offsets=tuple(offsets),
        kv_offsets=tuple(kv_offsets),
    )


def compact_by_segments(
    tensor: torch.Tensor,
    segments: Sequence[PrefillSuffixSegment],
    *,
    clone: bool = False,
) -> torch.Tensor:
    if not segments:
        out = tensor[:0]
    elif len(segments) == 1:
        seg = segments[0]
        out = tensor[seg.source_start : seg.source_start + seg.length]
    else:
        out = torch.cat(
            [tensor[s.source_start : s.source_start + s.length] for s in segments],
            dim=0,
        )
    out = out.contiguous()
    return out.clone() if clone else out


def compact_kv_indices_by_segments(
    kv_indices: torch.Tensor,
    segments: Sequence[PrefillSuffixSegment],
    *,
    batch_size: int,
) -> torch.Tensor:
    if not segments:
        return kv_indices[:0].contiguous()
    if len(segments) == batch_size and all(
        s.batch_index == i for i, s in enumerate(segments)
    ):
        return kv_indices.contiguous()
    if len(segments) == 1:
        seg = segments[0]
        return kv_indices[
            seg.kv_source_start : seg.kv_source_start + seg.kv_len
        ].contiguous()
    return torch.cat(
        [kv_indices[s.kv_source_start : s.kv_source_start + s.kv_len] for s in segments],
        dim=0,
    ).contiguous()
