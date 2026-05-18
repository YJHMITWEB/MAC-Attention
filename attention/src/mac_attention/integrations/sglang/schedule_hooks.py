from __future__ import annotations

from typing import Any, List, Optional

from .config import get_mac_config


def _get_mac_prefill_cache_update_starts(
    reqs: List[Any], prefix_lens: List[int], seq_lens: List[int]
) -> Optional[List[int]]:
    config = get_mac_config()
    if not config.enable_mac:
        return None
    capacity = int(config.mac_lookback_tokens_left or 0)
    if capacity <= 0:
        return None

    update_starts: List[int] = []
    for req, prefix_len, seq_len in zip(reqs, prefix_lens, seq_lens):
        final_prompt_len = len(req.origin_input_ids)
        window_start = max(0, final_prompt_len - capacity)
        update_start = max(prefix_len, window_start)
        update_starts.append(update_start if update_start < seq_len else -1)
    return update_starts


def schedule_prepare_for_extend_after(result: Any, self: Any, *args: Any, **kwargs: Any) -> Any:
    if hasattr(self, "prefix_lens") and hasattr(self, "seq_lens"):
        seq_lens = self.seq_lens
        if hasattr(seq_lens, "detach"):
            seq_lens = seq_lens.detach().cpu().tolist()
        self.mac_prefill_cache_update_starts = _get_mac_prefill_cache_update_starts(
            self.reqs,
            list(self.prefix_lens),
            [int(x) for x in seq_lens],
        )
    return result


def schedule_mix_with_running_after(
    result: Any, self: Any, running_batch: Any, *args: Any, **kwargs: Any
) -> Any:
    starts = getattr(self, "mac_prefill_cache_update_starts", None)
    if starts is not None:
        starts.extend([-1] * running_batch.batch_size())
    return result


def schedule_get_model_worker_batch_after(result: Any, self: Any, *args: Any, **kwargs: Any) -> Any:
    setattr(
        result,
        "mac_prefill_cache_update_starts",
        getattr(self, "mac_prefill_cache_update_starts", None),
    )
    return result


def forward_batch_init_new_after(result: Any, cls: Any, batch: Any, model_runner: Any) -> Any:
    setattr(
        result,
        "mac_prefill_cache_update_starts",
        getattr(batch, "mac_prefill_cache_update_starts", None),
    )
    return result
