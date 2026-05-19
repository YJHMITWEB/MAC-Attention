from __future__ import annotations

import inspect
import math
import os
import sys
from typing import Any, Optional

import torch

from .bridge import (
    load_mac_decode_rope_preserve_extension,
    load_mac_merge_downdate_cache_extension,
    load_mac_persistent_decode_extension,
    load_mac_prefill_update_cache_extension,
)
from .config import get_mac_config
from .prefill_suffix import (
    build_prefill_suffix_plan,
    compact_by_segments,
    compact_kv_indices_by_segments,
)
from .profiling import enabled as profiling_enabled
from .profiling import profile_block


def _graph_trace(message: str, layer: Any = None) -> None:
    if os.environ.get("MAC_DEBUG_GRAPH_TRACE", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    layer_id = getattr(layer, "layer_id", None)
    prefix = "[mac-graph-trace][flashinfer]"
    if layer_id is not None:
        prefix += f"[layer={layer_id}]"
    print(f"{prefix} {message}", file=sys.stderr, flush=True)


def flashinfer_backend_init_after(
    result: Any, self: Any, model_runner: Any, *args: Any, **kwargs: Any
) -> Any:
    server_args = getattr(model_runner, "server_args", None)
    config = get_mac_config(server_args)
    if not config.enable_mac:
        return result

    self.mac_config = config
    self.mac_prefill_kv_indices = [None for _ in range(self.num_wrappers)]
    self.mac_placeholder_attn_start_pos = None
    self.mac_has_retrieved = False
    self.mac_plan_total_pages_hint = int(
        getattr(
            model_runner,
            "max_total_num_tokens",
            getattr(getattr(model_runner, "server_args", None), "max_total_tokens", 0) or 0,
        )
        or 0
    )

    cuda_graph_enabled = not bool(getattr(server_args, "disable_cuda_graph", True))
    self.mac_cuda_graph_enabled = cuda_graph_enabled
    self.mac_sync_prefill_cache_update_when_cuda_graph = (
        cuda_graph_enabled
        and os.environ.get("MAC_SYNC_PREFILL_CACHE_UPDATE_WHEN_CUDA_GRAPH", "1")
        .strip()
        .lower()
        not in {"0", "false", "no", "off"}
    )
    default_async_slots = (
        1
        if cuda_graph_enabled
        else int(getattr(model_runner.model_config, "num_hidden_layers", 1))
    )
    async_slot_count = max(
        1,
        int(
            os.environ.get(
                "MAC_ASYNC_CACHE_UPDATE_SLOTS",
                str(default_async_slots),
            )
        ),
    )
    self.mac_async_cache_update_slots = []
    self.mac_async_cache_update_next_slot = 0
    self.mac_async_cache_update_deferred_count = 0

    if not getattr(self, "skip_prefill", False):
        defer_async_slots = cuda_graph_enabled and os.environ.get(
            "MAC_DEFER_ASYNC_CACHE_UPDATE_SLOTS", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        if defer_async_slots:
            self.mac_async_cache_update_deferred_count = async_slot_count
        else:
            for _ in range(async_slot_count):
                self.mac_async_cache_update_slots.append(
                    _make_async_cache_update_slot(self)
                )
    if cuda_graph_enabled and os.environ.get(
        "MAC_PRELOAD_EXTENSIONS_FOR_CUDA_GRAPH", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}:
        _preload_mac_extensions_for_cuda_graph(config)
    _mac_prefill_trace(
        "backend_init "
        f"enable_mac={config.enable_mac} "
        f"skip_prefill={getattr(self, 'skip_prefill', None)} "
        f"num_wrappers={getattr(self, 'num_wrappers', None)} "
        f"async_slots={len(getattr(self, 'mac_async_cache_update_slots', []) or [])}"
    )
    return result


def _preload_mac_extensions_for_cuda_graph(config: Any) -> None:
    verbose = os.environ.get("MAC_EXTENSION_VERBOSE", "0") == "1"
    _graph_trace("preload extensions enter")
    load_mac_prefill_update_cache_extension(verbose=verbose)
    _graph_trace("preload prefill_update_cache extension done")
    load_mac_merge_downdate_cache_extension(verbose=verbose)
    _graph_trace("preload merge_downdate_cache extension done")
    load_mac_persistent_decode_extension(
        verbose=verbose,
        fast_math=bool(config.mac_persistent_fast_math),
    )
    _graph_trace("preload persistent_decode extension done")
    load_mac_decode_rope_preserve_extension(verbose=verbose)
    _graph_trace("preload decode_rope_preserve extension done")


def _skip_flashinfer_replay_metadata_enabled() -> bool:
    return os.environ.get(
        "MAC_CUDAGRAPH_SKIP_FLASHINFER_REPLAY_METADATA", "1"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _decode_graph_state_for_backend(backend: Any) -> dict[int, dict[str, bool]]:
    states = getattr(backend, "mac_cudagraph_decode_states", None)
    if not isinstance(states, dict):
        states = {}
        backend.mac_cudagraph_decode_states = states
    return states


def _record_decode_graph_capture(
    backend: Any,
    forward_batch: Any,
    *,
    persistent: bool,
) -> None:
    if not _is_cuda_graph_capturing():
        return
    try:
        bs = int(getattr(forward_batch, "batch_size", 0))
    except Exception:
        return
    if bs <= 0:
        return
    state = _decode_graph_state_for_backend(backend).setdefault(
        bs,
        {"persistent_seen": False, "fallback_seen": False},
    )
    if persistent:
        state["persistent_seen"] = True
    else:
        state["fallback_seen"] = True


def flashinfer_init_forward_metadata_around(original_fn: Any, self: Any, forward_batch: Any) -> Any:
    with profile_block(
        "flashinfer.init_forward_metadata.original",
        forward_batch=forward_batch,
        cuda=False,
    ):
        result = original_fn(self, forward_batch)

    config = getattr(self, "mac_config", None)
    if config is None:
        config = get_mac_config()
    if config.enable_mac:
        # SGLang's piecewise CUDA graph replay creates a static ForwardBatch
        # and does not preserve plugin-only Python attributes. Keep the
        # scheduler's suffix-update gate on the backend so the FlashInfer hook
        # can make the same cache-update decision during PCG replay without
        # requiring a local SGLang patch.
        self.mac_prefill_cache_update_starts_current = getattr(
            forward_batch, "mac_prefill_cache_update_starts", None
        )
        self.mac_extend_prefix_lens_cpu_current = getattr(
            forward_batch, "extend_prefix_lens_cpu", None
        )
        self.mac_extend_seq_lens_cpu_current = getattr(
            forward_batch, "extend_seq_lens_cpu", None
        )
    if not config.enable_mac or not config.mac_force_paged_prefill:
        _mac_prefill_trace(
            "init_metadata skip config "
            f"enable_mac={config.enable_mac} "
            f"force_paged={config.mac_force_paged_prefill}"
        )
        return result
    if getattr(self, "skip_prefill", False) or self.num_wrappers != 1:
        _mac_prefill_trace(
            "init_metadata skip backend "
            f"skip_prefill={getattr(self, 'skip_prefill', None)} "
            f"num_wrappers={getattr(self, 'num_wrappers', None)}"
        )
        return result
    if (
        forward_batch.forward_mode.is_decode_or_idle()
        or forward_batch.forward_mode.is_draft_extend()
        or forward_batch.forward_mode.is_target_verify()
    ):
        _mac_prefill_trace(
            "init_metadata skip non-prefill "
            f"mode={getattr(forward_batch, 'forward_mode', None)}"
        )
        return result
    multi_item_params = getattr(
        getattr(self, "forward_metadata", None), "multi_item_params", None
    )
    if multi_item_params is not None and multi_item_params.is_enabled():
        _mac_prefill_trace("init_metadata skip multi_item")
        return result
    is_mixed = getattr(forward_batch.forward_mode, "is_mixed", lambda: False)()
    batch_size = int(getattr(forward_batch, "batch_size", 0))
    suffix_plan_for_graph = None
    try:
        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        q_len_for_plan = sum(max(0, int(x)) for x in extend_lens[:batch_size])
        suffix_plan_for_graph = build_prefill_suffix_plan(
            forward_batch, batch_size, q_len_for_plan
        )
        cache_update_required = (
            suffix_plan_for_graph is None or suffix_plan_for_graph.total_len > 0
        )
    except Exception:
        cache_update_required = True
    if not is_mixed and not cache_update_required:
        _mac_prefill_trace(
            "init_metadata skip no_cache_update "
            f"batch={getattr(forward_batch, 'batch_size', None)} "
            f"starts={getattr(forward_batch, 'mac_prefill_cache_update_starts', None)} "
            f"extend_lens={getattr(forward_batch, 'extend_seq_lens_cpu', None)} "
            f"prefix_lens={getattr(forward_batch, 'extend_prefix_lens_cpu', None)}"
        )
        return result
    update_kwargs = {
        "prefill_wrappers": self.prefill_wrappers_paged,
        "use_ragged": False,
        "encoder_lens": forward_batch.encoder_lens,
        "spec_info": None,
        "fixed_split_size": getattr(self, "prefill_split_tile_size", None),
    }
    try:
        update_params = inspect.signature(
            self.indices_updater_prefill.update
        ).parameters
    except (TypeError, ValueError):
        update_params = {}
    optional_kwargs = {
        "multi_item_params": multi_item_params,
        "cross_attention_custom_mask": getattr(
            forward_batch, "cross_attention_custom_mask", None
        ),
    }
    for name, value in optional_kwargs.items():
        if name in update_params:
            update_kwargs[name] = value

    with profile_block(
        "mac.force_paged_prefill_metadata",
        forward_batch=forward_batch,
        cuda=False,
    ):
        self.indices_updater_prefill.update(
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.seq_lens_cpu,
            forward_batch.seq_lens_sum,
            forward_batch.extend_prefix_lens,
            **update_kwargs,
        )
        metadata_type = type(self.forward_metadata)
        self.forward_metadata = metadata_type(
            self.prefill_wrappers_paged,
            False,
            False,
            multi_item_params,
        )
    _mac_prefill_trace(
        "init_metadata forced_paged "
        f"batch={getattr(forward_batch, 'batch_size', None)} "
        f"starts={getattr(forward_batch, 'mac_prefill_cache_update_starts', None)} "
        f"extend_lens={getattr(forward_batch, 'extend_seq_lens_cpu', None)} "
        f"prefix_lens={getattr(forward_batch, 'extend_prefix_lens_cpu', None)}"
    )
    return result


def flashinfer_init_forward_metadata_capture_cuda_graph_around(
    original_fn: Any, self: Any, *args: Any, **kwargs: Any
) -> Any:
    bs = args[0] if len(args) > 0 else kwargs.get("bs")
    forward_mode = args[5] if len(args) > 5 else kwargs.get("forward_mode")
    if _skip_flashinfer_replay_metadata_enabled():
        config = getattr(self, "mac_config", None)
        if config is None:
            config = get_mac_config()
        try:
            is_decode_graph = bool(forward_mode.is_decode_or_idle())
        except Exception:
            is_decode_graph = False
        if config.enable_mac and is_decode_graph and bs is not None:
            _decode_graph_state_for_backend(self)[int(bs)] = {
                "persistent_seen": False,
                "fallback_seen": False,
            }
    if os.environ.get("MAC_DEBUG_GRAPH_TRACE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        num_tokens = args[1] if len(args) > 1 else kwargs.get("num_tokens")
        _graph_trace(
            "capture_metadata enter "
            f"bs={bs} num_tokens={num_tokens} mode={forward_mode}"
        )
    result = original_fn(self, *args, **kwargs)
    _graph_trace("capture_metadata exit")
    return result


def flashinfer_init_forward_metadata_replay_cuda_graph_around(
    original_fn: Any, self: Any, *args: Any, **kwargs: Any
) -> Any:
    bs = args[0] if len(args) > 0 else kwargs.get("bs")
    forward_mode = args[5] if len(args) > 5 else kwargs.get("forward_mode")
    if _skip_flashinfer_replay_metadata_enabled() and bs is not None:
        config = getattr(self, "mac_config", None)
        if config is None:
            config = get_mac_config()
        try:
            is_decode_graph = bool(forward_mode.is_decode_or_idle())
        except Exception:
            is_decode_graph = False
        state = getattr(self, "mac_cudagraph_decode_states", {}).get(int(bs), {})
        if (
            config.enable_mac
            and is_decode_graph
            and bool(state.get("persistent_seen", False))
            and not bool(state.get("fallback_seen", False))
        ):
            _graph_trace(f"replay_metadata skip flashinfer bs={bs}")
            return None
    return original_fn(self, *args, **kwargs)


def _make_flashinfer_init_forward_metadata_direct(original_fn: Any) -> Any:
    def init_forward_metadata(self: Any, forward_batch: Any) -> Any:
        return flashinfer_init_forward_metadata_around(original_fn, self, forward_batch)

    return init_forward_metadata


flashinfer_init_forward_metadata_around.__mac_attention_direct_factory__ = (
    _make_flashinfer_init_forward_metadata_direct
)


def _supports_mac_backend(self: Any, layer: Any, forward_batch: Any) -> bool:
    config = getattr(layer, "mac_config", None)
    if config is None:
        config = get_mac_config()
    if not config.enable_mac:
        return False
    if self.num_wrappers != 1:
        return False
    if getattr(layer, "is_cross_attention", False):
        return False
    if getattr(layer, "sliding_window_size", -1) not in (-1, None):
        return False
    attn_type = getattr(getattr(layer, "attn_type", None), "value", "decoder")
    if attn_type != "decoder":
        return False
    if getattr(forward_batch, "spec_info", None) is not None:
        return False
    metadata = getattr(layer, "mac_metadata", None)
    if metadata is None or metadata.unified_query_cache is None:
        return False
    return True


def _build_prefill_kv_indices(backend: Any, forward_batch: Any, batch_size: int) -> torch.Tensor:
    from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton

    seq_lens = forward_batch.seq_lens[:batch_size]
    kv_indptr = backend.kv_indptr[0][: batch_size + 1]
    total = int(forward_batch.seq_lens_sum)
    kv_indices = torch.empty(total, dtype=torch.int32, device=seq_lens.device)
    create_flashinfer_kv_indices_triton[(batch_size,)](
        backend.indices_updater_prefill.req_to_token,
        forward_batch.req_pool_indices[:batch_size],
        seq_lens,
        kv_indptr,
        None,
        kv_indices,
        backend.indices_updater_prefill.req_to_token.shape[1],
    )
    backend.mac_prefill_kv_indices[0] = kv_indices
    return kv_indices


def _prefill_cache_update_required(
    forward_batch: Any, batch_size: int, q_len: int
) -> bool:
    suffix_plan = build_prefill_suffix_plan(forward_batch, batch_size, q_len)
    return suffix_plan is None or suffix_plan.total_len > 0


def _prefill_cache_update_required_for_batch(forward_batch: Any, batch_size: int) -> bool:
    extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    try:
        q_len = sum(max(0, int(x)) for x in extend_lens[:batch_size])
    except Exception:
        return True
    return _prefill_cache_update_required(forward_batch, batch_size, q_len)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _candidate_mode_id(mode: str) -> int:
    return 1 if str(mode).strip().lower() == "exclude_recent_right" else 0


def _mac_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        try:
            return bool(compiler.is_compiling())
        except Exception:
            return False
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        try:
            return bool(dynamo.is_compiling())
        except Exception:
            return False
    return False


def _is_cuda_graph_capturing() -> bool:
    if _is_torch_compiling():
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _is_piecewise_cuda_graph_active() -> bool:
    try:
        from sglang.srt.compilation.piecewise_context_manager import (
            is_in_piecewise_cuda_graph,
        )

        return bool(is_in_piecewise_cuda_graph())
    except Exception:
        return False


def _mac_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _bench_mode_id(mode: str) -> int:
    text = str(mode).strip().lower().replace("-", "_")
    if text == "synthetic_head":
        return 1
    if text == "synthetic_group":
        return 2
    return 0


def _persistent_debug_log_interval() -> int:
    raw = os.environ.get("MAC_PERSISTENT_DEBUG_LOG_INTERVAL", "0")
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _mac_prefill_trace_enabled(layer: Any = None) -> bool:
    raw = os.environ.get("MAC_PREFILL_TRACE", "0").strip().lower()
    if raw not in {"1", "true", "t", "yes", "y", "on"}:
        return False
    layer_filter = os.environ.get("MAC_PREFILL_TRACE_LAYERS", "").strip()
    if not layer_filter or layer is None:
        return True
    try:
        layer_id = int(getattr(layer, "layer_id", layer))
    except Exception:
        return True
    allowed = set()
    for item in layer_filter.replace(",", " ").split():
        try:
            allowed.add(int(item))
        except ValueError:
            pass
    return not allowed or layer_id in allowed


def _mac_prefill_trace(message: str, layer: Any = None) -> None:
    if not _mac_prefill_trace_enabled(layer):
        return
    layer_id = getattr(layer, "layer_id", layer)
    prefix = "[MAC_PREFILL_TRACE]"
    if layer_id is not None:
        prefix += f"[layer={layer_id}]"
    print(f"{prefix} {message}", flush=True)


def _log_persistent_debug_summary(
    backend: Any,
    workspace: dict[str, torch.Tensor],
    batch_size: int,
    layer: Any,
    config: Any,
) -> None:
    if _is_torch_compiling() or _is_cuda_graph_capturing() or _is_piecewise_cuda_graph_active():
        return
    interval = _persistent_debug_log_interval()
    if interval <= 0:
        return
    counter = int(getattr(backend, "mac_persistent_debug_log_counter", 0)) + 1
    backend.mac_persistent_debug_log_counter = counter
    if counter % interval != 0:
        return

    with torch.no_grad():
        head_hit = workspace["head_hit"][:batch_size]
        rect_begin = workspace["group_rect_begin"][:batch_size]
        rect_end = workspace["group_rect_end"][:batch_size]
        tail_begin = workspace["group_tail_begin"][:batch_size]
        tail_end = workspace["group_tail_end"][:batch_size]
        group_mode = workspace.get("group_mode")
        if group_mode is not None:
            group_mode = group_mode[:batch_size]
        group_rect_tiles = workspace["group_rect_tiles"][:batch_size]
        rect_tokens = (rect_end - rect_begin).clamp_min(0).float()
        tail_tokens = (tail_end - tail_begin).clamp_min(0).float()
        group_all_hit = group_mode == 0 if group_mode is not None else rect_begin > 0
        group_fallback = group_mode == 1 if group_mode is not None else rect_begin == 0
        group_mixed = group_mode == 2 if group_mode is not None else torch.zeros_like(group_all_hit)
        group_split = group_rect_tiles < 0
        head_rect_start = workspace["head_rect_start"][:batch_size]
        head_new_end = workspace["head_new_end"][:batch_size]
        group_size = max(1, head_hit.shape[-1] // max(1, rect_tokens.shape[-1]))
        head_hit_by_group = head_hit.reshape(batch_size, rect_tokens.shape[-1], group_size)
        head_tokens = (
            head_new_end - head_rect_start
        ).clamp_min(0).reshape(batch_size, rect_tokens.shape[-1], group_size).float()
        scheduled_head_slots = torch.where(
            group_split.unsqueeze(-1),
            head_tokens,
            rect_tokens.unsqueeze(-1).expand_as(head_tokens),
        )
        scheduled_tokens_per_group = scheduled_head_slots.sum(dim=-1) / float(group_size)
        actual_head_tokens = head_tokens.sum()
        unused_head_slots = (scheduled_head_slots - head_tokens).clamp_min(0).sum()
        unused_head_slot_ratio = (
            unused_head_slots / scheduled_head_slots.sum().clamp_min(1.0)
        ).item()
        mixed_miss_heads = (
            (~head_hit_by_group.bool())[group_mixed].float().mean().item()
            if group_mixed.any()
            else 0.0
        )
        mixed_scheduled_tokens = scheduled_tokens_per_group[group_mixed].sum()
        mixed_head_slots = scheduled_head_slots[group_mixed].sum()
        mixed_actual_head_tokens = (
            head_tokens[group_mixed].sum() if group_mixed.any() else rect_tokens.new_tensor(0.0)
        )
        mixed_unused_head_slots = (
            (scheduled_head_slots[group_mixed] - head_tokens[group_mixed]).clamp_min(0).sum()
            if group_mixed.any()
            else rect_tokens.new_tensor(0.0)
        )
        mixed_unused_head_slot_ratio = (
            mixed_unused_head_slots / mixed_head_slots.clamp_min(1.0)
        ).item()
        mixed_sched_token_share = (
            mixed_scheduled_tokens / scheduled_tokens_per_group.sum().clamp_min(1.0)
        ).item()
        miss_by_group = ~head_hit_by_group.bool()
        miss_count = miss_by_group.sum(dim=-1)
        miss_hist = torch.bincount(miss_count.reshape(-1).cpu(), minlength=group_size + 1)
        miss_hist_text = ",".join(
            f"{idx}:{int(count.item())}"
            for idx, count in enumerate(miss_hist)
            if int(count.item()) != 0
        )
        mixed_miss1_groups = int(((group_mixed) & (miss_count == 1)).sum().item())
        mixed_miss2_groups = int(((group_mixed) & (miss_count == 2)).sum().item())
        mixed_miss3_groups = int(((group_mixed) & (miss_count == 3)).sum().item())
        complete_group_rows = 0
        complete_head_rows = 0
        complete_mixed_z2_rows = 0
        complete_full_group_rows = 0
        complete_mixed_group_rows = 0
        complete_mac_hit_group_rows = 0
        estimated_partial_rect_states = 0
        estimated_active_partial_rect_states = 0
        estimated_inactive_partial_rect_states = 0
        fallback_reduce_groups = 0
        task_counts = workspace.get("task_counts")
        complete_task_go = workspace.get("complete_task_go")
        complete_task_tile = workspace.get("complete_task_tile")
        if task_counts is not None and complete_task_go is not None and complete_task_tile is not None:
            counts = task_counts[:9].detach().cpu()
            complete_rows = int(counts[0].item())
            fallback_reduce_groups = int(counts[3].item())
            if complete_rows > 0:
                task_go = complete_task_go[:complete_rows].detach().cpu()
                task_tile = complete_task_tile[:complete_rows].detach().cpu()
                group_total = int(batch_size * rect_tokens.shape[-1])
                task_base_go = task_go.clone()
                encoded_mixed_z2 = (task_go >= group_total) & (task_go < group_total * 2)
                task_base_go[encoded_mixed_z2] -= group_total
                task_is_group = task_go >= 0
                task_valid_group = (
                    task_is_group & (task_base_go >= 0) & (task_base_go < group_total)
                )
                group_mode_flat = group_mode.reshape(-1).detach().cpu()
                miss_count_flat = miss_count.reshape(-1).detach().cpu()
                task_mode = torch.full_like(task_go, -1)
                task_miss_count = torch.zeros_like(task_go)
                if task_valid_group.any():
                    task_mode[task_valid_group] = group_mode_flat[
                        task_base_go[task_valid_group]
                    ]
                    task_miss_count[task_valid_group] = miss_count_flat[
                        task_base_go[task_valid_group]
                    ].to(task_miss_count.dtype)
                complete_group_rows = int((task_go >= 0).sum().item())
                complete_head_rows = int((task_go < 0).sum().item())
                complete_mixed_z2_rows = int(encoded_mixed_z2.sum().item())
                complete_full_group_rows = int(
                    (task_valid_group & (task_mode == 1)).sum().item()
                )
                complete_mixed_group_rows = int(
                    (task_valid_group & (task_mode == 2) & (~encoded_mixed_z2)).sum().item()
                )
                complete_mac_hit_group_rows = int(
                    (task_valid_group & (task_mode == 0)).sum().item()
                )
                estimated_partial_rect_states = complete_head_rows + complete_group_rows * group_size
                estimated_active_partial_rect_states = complete_head_rows
                if task_valid_group.any():
                    rect_begin_flat = rect_begin.reshape(-1).detach().cpu()
                    rect_end_flat = rect_end.reshape(-1).detach().cpu()
                    head_rect_start_by_group = head_rect_start.reshape(
                        batch_size, rect_tokens.shape[-1], group_size
                    ).detach().cpu()
                    head_new_end_by_group = head_new_end.reshape(
                        batch_size, rect_tokens.shape[-1], group_size
                    ).detach().cpu()
                    group_task_tile_counts = torch.zeros(group_total, dtype=torch.int64)
                    for go_i in task_base_go[task_valid_group].tolist():
                        if 0 <= int(go_i) < group_total:
                            group_task_tile_counts[int(go_i)] += 1
                    for go_i in torch.nonzero(
                        group_task_tile_counts, as_tuple=False
                    ).flatten().tolist():
                        group_task_tile_counts[go_i] = max(
                            int(group_task_tile_counts[go_i].item()),
                            int(task_tile[task_valid_group & (task_base_go == go_i)].max().item())
                            + 1,
                        )
                    for idx in torch.nonzero(task_valid_group, as_tuple=False).flatten().tolist():
                        go_i = int(task_base_go[idx].item())
                        mode_i = int(task_mode[idx].item())
                        if mode_i in (0, 1):
                            estimated_active_partial_rect_states += group_size
                            continue
                        tile_count = max(int(group_task_tile_counts[go_i].item()), 1)
                        rect_tokens_i = max(
                            int(rect_end_flat[go_i].item() - rect_begin_flat[go_i].item()),
                            0,
                        )
                        tile_tokens_i = max((rect_tokens_i + tile_count - 1) // tile_count, 1)
                        begin_i = (
                            int(rect_begin_flat[go_i].item())
                            + int(task_tile[idx].item()) * tile_tokens_i
                        )
                        end_i = min(begin_i + tile_tokens_i, int(rect_end_flat[go_i].item()))
                        n_i = go_i // rect_tokens.shape[-1]
                        hkv_i = go_i % rect_tokens.shape[-1]
                        active_heads = 0
                        for lane in range(group_size):
                            head_begin_i = int(
                                head_rect_start_by_group[n_i, hkv_i, lane].item()
                            )
                            head_end_i = int(
                                head_new_end_by_group[n_i, hkv_i, lane].item()
                            )
                            if end_i > head_begin_i and begin_i < head_end_i:
                                active_heads += 1
                        estimated_active_partial_rect_states += active_heads
                estimated_inactive_partial_rect_states = max(
                    estimated_partial_rect_states - estimated_active_partial_rect_states,
                    0,
                )
        partial_rect_o = workspace.get("partial_rect_o")
        partial_state_bytes = 0
        if partial_rect_o is not None:
            partial_state_bytes = int(partial_rect_o.shape[-1]) * int(partial_rect_o.element_size()) + 4
        estimated_partial_rect_mb = (
            estimated_partial_rect_states * partial_state_bytes / float(1024 * 1024)
        )
        estimated_phase4_rect_state_reads = int(group_rect_tiles.clamp_min(0).sum().item() * group_size)
        estimated_active_phase4_rect_state_reads = 0
        rect_tiles_flat = group_rect_tiles.reshape(-1).detach().cpu()
        rect_begin_flat = rect_begin.reshape(-1).detach().cpu()
        rect_end_flat = rect_end.reshape(-1).detach().cpu()
        group_mode_flat_for_reads = group_mode.reshape(-1).detach().cpu()
        head_rect_start_by_group = head_rect_start.reshape(
            batch_size, rect_tokens.shape[-1], group_size
        ).detach().cpu()
        head_new_end_by_group = head_new_end.reshape(
            batch_size, rect_tokens.shape[-1], group_size
        ).detach().cpu()
        group_total = int(batch_size * rect_tokens.shape[-1])
        for go_i in range(group_total):
            tiles_i = int(max(int(rect_tiles_flat[go_i].item()), 0))
            if tiles_i <= 0:
                continue
            mode_i = int(group_mode_flat_for_reads[go_i].item())
            if mode_i in (0, 1):
                estimated_active_phase4_rect_state_reads += tiles_i * group_size
                continue
            rect_tokens_i = max(
                int(rect_end_flat[go_i].item() - rect_begin_flat[go_i].item()),
                0,
            )
            tile_tokens_i = max((rect_tokens_i + tiles_i - 1) // tiles_i, 1)
            n_i = go_i // rect_tokens.shape[-1]
            hkv_i = go_i % rect_tokens.shape[-1]
            for tile_i in range(tiles_i):
                begin_i = int(rect_begin_flat[go_i].item()) + tile_i * tile_tokens_i
                end_i = min(begin_i + tile_tokens_i, int(rect_end_flat[go_i].item()))
                for lane in range(group_size):
                    head_begin_i = int(head_rect_start_by_group[n_i, hkv_i, lane].item())
                    head_end_i = int(head_new_end_by_group[n_i, hkv_i, lane].item())
                    if end_i > head_begin_i and begin_i < head_end_i:
                        estimated_active_phase4_rect_state_reads += 1
        estimated_inactive_phase4_rect_state_reads = max(
            estimated_phase4_rect_state_reads - estimated_active_phase4_rect_state_reads,
            0,
        )
        estimated_phase4_rect_read_mb = (
            estimated_phase4_rect_state_reads * partial_state_bytes / float(1024 * 1024)
        )
        inactive_partial_state_ratio = (
            estimated_inactive_partial_rect_states / max(estimated_partial_rect_states, 1)
        )
        inactive_phase4_read_ratio = (
            estimated_inactive_phase4_rect_state_reads
            / max(estimated_phase4_rect_state_reads, 1)
        )
        cache_len_max = int((tail_end.max() - 1).item()) if tail_end.numel() else 0
        layer_id = getattr(layer, "layer_id", None)
        print(
            "[MAC persistent] "
            f"layer={layer_id} "
            f"cache_len_max={cache_len_max} "
            f"head_hit={head_hit.float().mean().item():.4f} "
            f"group_all_hit={group_all_hit.float().mean().item():.4f} "
            f"group_fallback={group_fallback.float().mean().item():.4f} "
            f"group_mixed={group_mixed.float().mean().item():.4f} "
            f"group_split={group_split.float().mean().item():.4f} "
            f"complete_mean={scheduled_tokens_per_group.mean().item():.1f} "
            f"complete_max={scheduled_tokens_per_group.max().item():.0f} "
            f"tail_mean={tail_tokens.mean().item():.1f} "
            f"complete_actual_head_tokens={actual_head_tokens.item():.0f} "
            f"complete_unused_head_slot_ratio={unused_head_slot_ratio:.4f} "
            f"mixed_miss_heads_mean={mixed_miss_heads:.2f} "
            f"mixed_sched_token_share={mixed_sched_token_share:.4f} "
            f"mixed_unused_head_slot_ratio={mixed_unused_head_slot_ratio:.4f} "
            f"miss_hist={miss_hist_text} "
            f"mixed_miss1_groups={mixed_miss1_groups} "
            f"mixed_miss2_groups={mixed_miss2_groups} "
            f"mixed_miss3_groups={mixed_miss3_groups} "
            f"complete_group_rows={complete_group_rows} "
            f"complete_head_rows={complete_head_rows} "
            f"complete_mixed_z2_rows={complete_mixed_z2_rows} "
            f"complete_full_group_rows={complete_full_group_rows} "
            f"complete_mixed_group_rows={complete_mixed_group_rows} "
            f"complete_mac_hit_group_rows={complete_mac_hit_group_rows} "
            f"estimated_partial_rect_states={estimated_partial_rect_states} "
            f"estimated_active_partial_rect_states={estimated_active_partial_rect_states} "
            f"estimated_inactive_partial_rect_states={estimated_inactive_partial_rect_states} "
            f"estimated_inactive_partial_rect_state_ratio={inactive_partial_state_ratio:.4f} "
            f"estimated_partial_rect_MB={estimated_partial_rect_mb:.3f} "
            f"estimated_phase4_rect_state_reads={estimated_phase4_rect_state_reads} "
            f"estimated_active_phase4_rect_state_reads={estimated_active_phase4_rect_state_reads} "
            f"estimated_inactive_phase4_rect_state_reads={estimated_inactive_phase4_rect_state_reads} "
            f"estimated_inactive_phase4_rect_read_ratio={inactive_phase4_read_ratio:.4f} "
            f"estimated_phase4_rect_read_MB={estimated_phase4_rect_read_mb:.3f}",
            flush=True,
        )
        if int(getattr(config, "mac_persistent_debug", 0) or 0) >= 2:
            phase_cycles = workspace.get("phase_cycles")
            if phase_cycles is not None:
                cycles = [int(x) for x in phase_cycles.detach().cpu().tolist()]
                if len(cycles) >= 6:
                    names = ("match", "reduce", "complete", "tail", "merge")
                    deltas = [max(cycles[i + 1] - cycles[i], 0) for i in range(5)]
                    total_cycles = sum(deltas)
                    if total_cycles > 0:
                        parts = " ".join(
                            f"{name}={delta / total_cycles * 100.0:.1f}%"
                            for name, delta in zip(names, deltas)
                        )
                        detail_parts = ""
                        if len(cycles) >= 13 and cycles[11] > cycles[0]:
                            detail_total = max(cycles[11] - cycles[0], 0)
                            detail = []
                            tail_before_reduce = (
                                len(cycles) >= 14
                                and cycles[13] >= cycles[6]
                                and cycles[7] >= cycles[13]
                                and cycles[8] >= cycles[7]
                            )
                            detail.append(
                                "phase_order="
                                + ("tail_before_reduce" if tail_before_reduce else "reduce_before_tail")
                            )
                            detail_ranges = [
                                ("prepare", 1, 2),
                                ("complete", 2, 6),
                            ]
                            if tail_before_reduce:
                                detail_ranges.extend(
                                    [
                                        ("tail", 6, 13),
                                        ("fallback_group_merge", 8, 9),
                                    ]
                                )
                            else:
                                detail_ranges.extend(
                                    [
                                        ("tail", 7, 8),
                                        ("fallback_group_merge", 8, 9),
                                    ]
                                )
                            detail_ranges.extend(
                                [
                                    ("merge_write", 9, 10),
                                    ("cache_subtract", 10, 11),
                                ]
                            )
                            for name, i, j in detail_ranges:
                                detail.append(
                                    f"{name}={max(cycles[j] - cycles[i], 0) / detail_total * 100.0:.1f}%"
                                )
                            reduce_sync = 0
                            reduce_compute = 0
                            if (
                                fallback_reduce_groups > 0
                                and cycles[12] >= (cycles[13] if tail_before_reduce else cycles[6])
                                and cycles[7] >= cycles[12]
                            ):
                                reduce_begin_wait_base = cycles[13] if tail_before_reduce else cycles[6]
                                reduce_sync = cycles[12] - reduce_begin_wait_base
                                reduce_compute = cycles[7] - cycles[12]
                            detail.append(
                                f"reduce_sync_wait={reduce_sync / detail_total * 100.0:.1f}%"
                            )
                            detail.append(
                                f"reduce_compute={reduce_compute / detail_total * 100.0:.1f}%"
                            )
                            detail_parts = " detail_" + " detail_".join(detail)
                        print(
                            f"[MAC persistent phases] layer={layer_id} "
                            f"total_cycles={total_cycles} {parts}{detail_parts}",
                            flush=True,
                        )


def _update_persistent_decode_profile(
    metadata: Any,
    workspace: dict[str, torch.Tensor],
    batch_size: int,
    past_lens: torch.Tensor,
) -> None:
    with torch.no_grad():
        head_hit = workspace["head_hit"][:batch_size].bool()
        head_prefix_end = workspace["head_prefix_end"][:batch_size].clamp_min(0)
        total_kv = past_lens[:batch_size].view(batch_size, 1).expand_as(head_prefix_end)
        hit_count = int(head_hit.sum().item())
        head_count = int(head_hit.numel())
        skipped = torch.where(
            head_hit,
            head_prefix_end,
            torch.zeros_like(head_prefix_end),
        )

        metadata.decoding_hit += hit_count
        metadata.decoding_miss += head_count - hit_count
        metadata.decoding_total_kv += int(total_kv.sum().item())
        metadata.decoding_skipped_kv += int(skipped.sum().item())

        if hit_count > 0:
            fast_range = (total_kv - head_prefix_end).clamp_min(0)
            metadata.decoding_range.append(float(fast_range[head_hit].float().mean().item()))

        continuous_hit = getattr(metadata, "continuous_hit", None)
        if torch.is_tensor(continuous_hit) and head_hit.numel() > 0:
            record_max_hit = head_hit.to(dtype=continuous_hit.dtype).max(dim=0).values
            metadata.continuous_hit += record_max_hit.to(
                device=continuous_hit.device,
                dtype=continuous_hit.dtype,
                non_blocking=True,
            )


def _maybe_eject_low_hit_persistent_cache(
    metadata: Any,
    workspace: dict[str, torch.Tensor],
    batch_size: int,
    req_ids: torch.Tensor,
    past_lens: torch.Tensor,
) -> None:
    if _is_cuda_graph_capturing():
        return
    threshold = _mac_float_env("MAC_ADAPTIVE_LOW_HIT_EJECT_THRESHOLD", -1.0)
    if threshold < 0.0:
        return
    threshold = max(0.0, min(1.0, threshold))
    min_kv = max(0, _mac_int_env("MAC_ADAPTIVE_LOW_HIT_EJECT_MIN_KV", 0))
    if min_kv > 0 and int(past_lens[:batch_size].min().item()) < min_kv:
        return
    group_mode = workspace.get("group_mode")
    if group_mode is not None:
        group_mode_now = group_mode[:batch_size]
        total_groups = int(group_mode_now.numel())
        non_hit_groups = int((group_mode_now != 0).sum().item())
    else:
        task_counts = workspace.get("task_counts")
        if task_counts is None:
            return
        Hkv = int(getattr(metadata, "num_kv_heads", 0) or getattr(metadata, "num_k_head", 0) or 0)
        total_groups = batch_size * Hkv
        if total_groups <= 0:
            return
        counts = task_counts[:9].detach().cpu()
        non_hit_groups = int(counts[5].item())
    if total_groups <= 0:
        return
    non_hit_ratio = non_hit_groups / float(total_groups)
    if non_hit_ratio >= threshold:
        if os.environ.get("MAC_ADAPTIVE_LOW_HIT_EJECT_LOG", "0") == "1":
            print(
                "[MAC adaptive] eject persistent cache "
                f"non_hit_groups={non_hit_groups}/{total_groups} "
                f"threshold={threshold:.3f}",
                flush=True,
            )
        _set_low_hit_ejected(metadata, req_ids, True)
        _invalidate_persistent_ready(metadata, req_ids)


def _ensure_persistent_workspace(backend: Any, metadata: Any, config: Any, batch_size: int) -> dict[str, torch.Tensor]:
    device = metadata.device
    assert device is not None
    max_running = int(
        max(
            batch_size,
            getattr(config, "max_running_requests", 0) or 0,
            metadata.unified_query_cache.shape[0],
        )
    )
    Hq = int(metadata.num_heads)
    Hkv = int(getattr(metadata, "num_kv_heads", 0) or getattr(metadata, "num_k_head", 0))
    D = int(metadata.head_dim)
    group = Hq // Hkv
    M = int(config.mac_lookback_tokens_left)
    tile = int(config.mac_persistent_tile_tokens)
    match_tile = int(config.mac_persistent_match_tile_slots)
    max_context = int(config.mac_persistent_max_context)
    max_match_tiles = _ceil_div(M, match_tile)
    max_tiles_context = _ceil_div(max_context, tile)
    max_tiles_reduce = _ceil_div(max_context, tile * 32)
    max_tiles_tail = max(1, _ceil_div(max(int(config.mac_semantic_pos_ahead), 1), tile))
    partial_o_dtype = torch.float32 if bool(config.mac_persistent_partial_fp32) else torch.bfloat16
    key = (
        device,
        max_running,
        Hq,
        Hkv,
        D,
        group,
        M,
        max_context,
        tile,
        match_tile,
        max_match_tiles,
        max_tiles_context,
        max_tiles_reduce,
        max_tiles_tail,
        partial_o_dtype,
    )
    current_key = getattr(backend, "mac_persistent_workspace_key", None)
    workspace = getattr(backend, "mac_persistent_workspace", None)
    if workspace is not None and current_key == key:
        return workspace

    i32 = torch.int32
    workspace = {
        "match_dist": torch.empty((max_running, Hq, max_match_tiles), device=device, dtype=torch.float32),
        "match_pos": torch.empty((max_running, Hq, max_match_tiles), device=device, dtype=i32),
        "match_slot": torch.empty((max_running, Hq, max_match_tiles), device=device, dtype=i32),
        "head_hit": torch.empty((max_running, Hq), device=device, dtype=i32),
        "head_match_slot": torch.empty((max_running, Hq), device=device, dtype=i32),
        "head_match_pos": torch.empty((max_running, Hq), device=device, dtype=i32),
        "head_prefix_end": torch.empty((max_running, Hq), device=device, dtype=i32),
        "head_rect_start": torch.empty((max_running, Hq), device=device, dtype=i32),
        "head_new_end": torch.empty((max_running, Hq), device=device, dtype=i32),
        "group_rect_begin": torch.empty((max_running, Hkv), device=device, dtype=i32),
        "group_rect_end": torch.empty((max_running, Hkv), device=device, dtype=i32),
        "group_rect_tiles": torch.empty((max_running, Hkv), device=device, dtype=i32),
        "group_tail_begin": torch.empty((max_running, Hkv), device=device, dtype=i32),
        "group_tail_end": torch.empty((max_running, Hkv), device=device, dtype=i32),
        "group_tail_tiles": torch.empty((max_running, Hkv), device=device, dtype=i32),
        "group_mode": torch.empty((max_running, Hkv), device=device, dtype=i32),
        "complete_task_go": torch.empty(
            (max_running * Hq * max_tiles_context,), device=device, dtype=i32
        ),
        "complete_task_tile": torch.empty(
            (max_running * Hq * max_tiles_context,), device=device, dtype=i32
        ),
        "tail_task_go": torch.empty(
            (max_running * Hkv * max_tiles_tail,), device=device, dtype=i32
        ),
        "tail_task_tile": torch.empty(
            (max_running * Hkv * max_tiles_tail,), device=device, dtype=i32
        ),
        "hit_tail_task_ho": torch.empty(
            (max_running * Hq * max_tiles_tail,), device=device, dtype=i32
        ),
        "hit_tail_task_tile": torch.empty(
            (max_running * Hq * max_tiles_tail,), device=device, dtype=i32
        ),
        "task_counts": torch.empty((9,), device=device, dtype=i32),
        "partial_rect_o": torch.empty(
            (max_running, Hkv, max_tiles_context, group, D), device=device, dtype=partial_o_dtype
        ),
        "partial_rect_lse": torch.empty(
            (max_running, Hkv, max_tiles_context, group), device=device, dtype=torch.float32
        ),
        "partial_rect_reduced_o": torch.empty(
            (max_running, Hkv, max_tiles_reduce, group, D), device=device, dtype=partial_o_dtype
        ),
        "partial_rect_reduced_lse": torch.empty(
            (max_running, Hkv, max_tiles_reduce, group), device=device, dtype=torch.float32
        ),
        "partial_tail_o": torch.empty(
            (max_running, Hkv, max_tiles_tail, group, D), device=device, dtype=partial_o_dtype
        ),
        "partial_tail_lse": torch.empty(
            (max_running, Hkv, max_tiles_tail, group), device=device, dtype=torch.float32
        ),
        "phase_cycles": torch.empty((14,), device=device, dtype=torch.int64),
    }
    backend.mac_persistent_workspace_key = key
    backend.mac_persistent_workspace = workspace
    return workspace


def _req_ids_to_host_list(metadata: Any, req_ids: torch.Tensor) -> list[int]:
    if _is_torch_compiling() or _is_cuda_graph_capturing() or _is_piecewise_cuda_graph_active():
        req_ids_host = getattr(metadata, "req_ids_host", None)
        if req_ids_host is None:
            return []
        return [int(x) for x in req_ids_host[: int(req_ids.numel())]]
    try:
        return [int(x) for x in req_ids.detach().cpu().tolist()]
    except Exception:
        req_ids_host = getattr(metadata, "req_ids_host", None)
        if req_ids_host is None:
            return []
        return [int(x) for x in req_ids_host[: int(req_ids.numel())]]


def _ensure_low_hit_ejected_host(metadata: Any) -> Optional[list[bool]]:
    ready_host = getattr(metadata, "mac_cache_ready_host", None)
    if ready_host is None:
        return None
    ejected = getattr(metadata, "mac_low_hit_ejected_host", None)
    if ejected is None:
        ejected = [False] * len(ready_host)
        metadata.mac_low_hit_ejected_host = ejected
    elif len(ejected) < len(ready_host):
        ejected.extend([False] * (len(ready_host) - len(ejected)))
    return ejected


def _set_low_hit_ejected(metadata: Any, req_ids: torch.Tensor, value: bool) -> None:
    ejected = _ensure_low_hit_ejected_host(metadata)
    if ejected is None:
        return
    for req in _req_ids_to_host_list(metadata, req_ids):
        if 0 <= req < len(ejected):
            ejected[req] = value


def _has_low_hit_ejected(metadata: Any, req_ids: torch.Tensor) -> bool:
    if _is_cuda_graph_capturing() or _is_piecewise_cuda_graph_active():
        return False
    ejected = getattr(metadata, "mac_low_hit_ejected_host", None)
    if ejected is None:
        return False
    for req in _req_ids_to_host_list(metadata, req_ids):
        if req < 0 or req >= len(ejected) or ejected[req]:
            return True
    return False


def _mark_persistent_ready(metadata: Any, req_ids: torch.Tensor) -> None:
    if _has_low_hit_ejected(metadata, req_ids):
        return
    ready_host = getattr(metadata, "mac_cache_ready_host", None)
    if ready_host is not None:
        for req in _req_ids_to_host_list(metadata, req_ids):
            if 0 <= int(req) < len(ready_host):
                ready_host[int(req)] = True
    if (
        getattr(metadata, "mac_cache_ready", None) is None
        or getattr(metadata, "mac_cache_epoch", None) is None
        or getattr(metadata, "request_epoch", None) is None
    ):
        return
    if _is_cuda_graph_capturing() or _is_piecewise_cuda_graph_active() or _is_torch_compiling():
        mark_ready = getattr(getattr(metadata, "cache_ext", None), "mac_mark_persistent_ready", None)
        if mark_ready is not None:
            mark_ready(
                metadata.mac_cache_ready,
                metadata.mac_cache_epoch,
                metadata.request_epoch,
                req_ids,
            )
            return
    reqs_host = _req_ids_to_host_list(metadata, req_ids)
    if reqs_host:
        for req in reqs_host:
            if req < 0 or req >= metadata.mac_cache_ready.shape[0]:
                continue
            metadata.mac_cache_epoch[req].copy_(metadata.request_epoch[req])
            metadata.mac_cache_ready[req].fill_(True)
        return
    req_long = req_ids.long()
    metadata.mac_cache_epoch[req_long] = metadata.request_epoch[req_long]
    metadata.mac_cache_ready[req_long] = True


def _invalidate_persistent_ready(metadata: Any, req_ids: torch.Tensor) -> None:
    ready_host = getattr(metadata, "mac_cache_ready_host", None)
    if ready_host is not None:
        for req in _req_ids_to_host_list(metadata, req_ids):
            if 0 <= int(req) < len(ready_host):
                ready_host[int(req)] = False
    ready = getattr(metadata, "mac_cache_ready", None)
    epoch = getattr(metadata, "mac_cache_epoch", None)
    if ready is None or epoch is None:
        return
    if (
        getattr(metadata, "request_epoch", None) is not None
        and (_is_cuda_graph_capturing() or _is_piecewise_cuda_graph_active() or _is_torch_compiling())
    ):
        invalidate_ready = getattr(
            getattr(metadata, "cache_ext", None),
            "mac_invalidate_persistent_ready",
            None,
        )
        if invalidate_ready is not None:
            invalidate_ready(
                ready,
                epoch,
                metadata.request_epoch,
                req_ids,
            )
            return
    reqs_host = _req_ids_to_host_list(metadata, req_ids)
    if reqs_host:
        for req in reqs_host:
            if req < 0 or req >= ready.shape[0]:
                continue
            ready[req].fill_(False)
            epoch[req].fill_(-1)
        return
    req_long = req_ids.long()
    ready[req_long] = False
    epoch[req_long] = -1


def _persistent_ready(metadata: Any, req_ids: torch.Tensor) -> bool:
    if _is_cuda_graph_capturing():
        return True
    if _has_low_hit_ejected(metadata, req_ids):
        return False
    ready_host = getattr(metadata, "mac_cache_ready_host", None)
    req_ids_host = getattr(metadata, "req_ids_host", None)
    if ready_host is not None and req_ids_host is not None:
        count = int(req_ids.numel())
        for req in req_ids_host[:count]:
            req_i = int(req)
            if req_i < 0 or req_i >= len(ready_host) or not ready_host[req_i]:
                return False
        return True
    ready = getattr(metadata, "mac_cache_ready", None)
    cache_epoch = getattr(metadata, "mac_cache_epoch", None)
    request_epoch = getattr(metadata, "request_epoch", None)
    if ready is None or cache_epoch is None or request_epoch is None:
        return False
    req_long = req_ids.long()
    # This support check is outside the persistent hot path. It may synchronize, but it is
    # required to avoid reading stale or legacy-semantics cache rows.
    return bool(torch.all(ready[req_long] & (cache_epoch[req_long] == request_epoch[req_long])).item())


def _try_persistent_decode(
    backend: Any,
    forward_batch: Any,
    config: Any,
    metadata: Any,
    q_input: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    cache_loc: torch.Tensor,
    layer: Any,
) -> Optional[torch.Tensor]:
    if not config.mac_persistent_coop:
        return None
    if metadata.persistent_decode_ext is None:
        return None
    if str(config.mac_persistent_cache_layout).strip().lower() != "slot_major":
        return None
    if str(config.mac_persistent_candidate_mode).strip().lower() not in {
        "last_m",
        "exclude_recent_right",
    }:
        return None
    if q_input.dtype is not torch.bfloat16 or k_buffer.dtype is not torch.bfloat16:
        return None
    if q_input.dim() != 3 or k_buffer.dim() != 3 or v_buffer.dim() != 3:
        return None
    batch_size = int(forward_batch.batch_size)
    if q_input.shape[0] != batch_size or batch_size <= 0:
        return None
    if int(q_input.shape[2]) != 128:
        return None
    max_kv_len = int(getattr(metadata, "cur_kv_len", 0) or 0)
    if max_kv_len > int(config.mac_persistent_max_context):
        return None
    Hq = int(q_input.shape[1])
    Hkv = int(k_buffer.shape[1])
    if Hkv <= 0 or Hq % Hkv != 0 or Hq // Hkv > 8:
        return None
    if int(getattr(layer, "logit_cap", 0) or 0) != 0:
        return None
    if getattr(layer, "k_scale_float", None) is not None or getattr(layer, "v_scale_float", None) is not None:
        return None
    req_to_token = getattr(backend.indices_updater_decode, "req_to_token", None)
    if not torch.is_tensor(req_to_token) or req_to_token.dtype != torch.int32 or not req_to_token.is_contiguous():
        return None
    req_ids = metadata.req_ids[:batch_size]
    if req_ids.dtype != torch.int32 or not req_ids.is_contiguous():
        return None
    past_lens = metadata.cache_lens[:batch_size]
    if past_lens.dtype != torch.int32 or not past_lens.is_contiguous():
        return None
    if (
        not torch.is_tensor(cache_loc)
        or cache_loc.dtype not in (torch.int32, torch.int64)
        or not cache_loc.is_contiguous()
    ):
        return None
    if not _persistent_ready(metadata, req_ids):
        if _mac_prefill_trace_enabled(layer):
            counter = int(getattr(backend, "mac_prefill_trace_decode_not_ready", 0))
            if counter < int(os.environ.get("MAC_PREFILL_TRACE_DECODE_LIMIT", "8")):
                ready_host = getattr(metadata, "mac_cache_ready_host", None)
                req_ids_host = getattr(metadata, "req_ids_host", None)
                _mac_prefill_trace(
                    "decode persistent_not_ready "
                    f"req_ids={req_ids_host[:batch_size] if req_ids_host is not None else None} "
                    f"ready_host={ready_host[:min(len(ready_host), 4)] if ready_host is not None else None} "
                    f"past_lens={past_lens.detach().cpu().tolist() if torch.is_tensor(past_lens) else past_lens}",
                    layer,
                )
            backend.mac_prefill_trace_decode_not_ready = counter + 1
        return None

    if not q_input.is_contiguous():
        q_input = q_input.contiguous()
    q_to_cache = metadata.cur_q_before_rope[:batch_size]
    if not q_to_cache.is_contiguous():
        q_to_cache = q_to_cache.contiguous()

    workspace = _ensure_persistent_workspace(backend, metadata, config, batch_size)
    out = torch.empty_like(q_input)
    optional_lse = torch.empty((0,), device=q_input.device, dtype=torch.float32)
    metadata.persistent_decode_ext.mac_persistent_decode_bf16(
        q_input,
        q_to_cache,
        k_buffer,
        v_buffer,
        req_to_token,
        req_ids,
        past_lens,
        cache_loc,
        metadata.unified_query_cache,
        metadata.unified_attn_cache,
        metadata.unified_lse_cache,
        out,
        optional_lse,
        workspace["match_dist"],
        workspace["match_pos"],
        workspace["match_slot"],
        workspace["head_hit"],
        workspace["head_match_slot"],
        workspace["head_match_pos"],
        workspace["head_prefix_end"],
        workspace["head_rect_start"],
        workspace["head_new_end"],
        workspace["group_rect_begin"],
        workspace["group_rect_end"],
        workspace["group_rect_tiles"],
        workspace["group_tail_begin"],
        workspace["group_tail_end"],
        workspace["group_tail_tiles"],
        workspace["group_mode"],
        workspace["complete_task_go"],
        workspace["complete_task_tile"],
        workspace["tail_task_go"],
        workspace["tail_task_tile"],
        workspace["hit_tail_task_ho"],
        workspace["hit_tail_task_tile"],
        workspace["task_counts"],
        workspace["partial_rect_o"],
        workspace["partial_rect_lse"],
        workspace["partial_rect_reduced_o"],
        workspace["partial_rect_reduced_lse"],
        workspace["partial_tail_o"],
        workspace["partial_tail_lse"],
        workspace["phase_cycles"],
        int(config.mac_persistent_max_context),
        int(config.mac_persistent_tile_tokens),
        int(config.mac_persistent_match_tile_slots),
        int(config.mac_semantic_pos_ahead),
        int(config.mac_gen_min_limit),
        int(config.mac_lookback_tokens_right),
        _candidate_mode_id(config.mac_persistent_candidate_mode),
        float(config.mac_threshold),
        float(layer.scaling),
        int(config.mac_persistent_debug),
        _bench_mode_id(getattr(config, "mac_bench_mode", "off")),
        float(getattr(config, "mac_bench_hit_rate", 1.0)),
        float(getattr(config, "mac_bench_hit_rate_std", 0.0)),
        float(getattr(config, "mac_bench_skip_ratio", -1.0)),
        float(getattr(config, "mac_bench_skip_ratio_std", 0.0)),
        float(getattr(config, "mac_bench_match_lag_mean", 64.0)),
        float(getattr(config, "mac_bench_match_lag_std", 0.0)),
        int(getattr(config, "mac_bench_seed", 1)),
        -1,
        int(getattr(layer, "layer_id", 0)),
    )
    if int(getattr(config, "mac_profile", 0) or 0):
        _update_persistent_decode_profile(metadata, workspace, batch_size, past_lens)
    _log_persistent_debug_summary(backend, workspace, batch_size, layer, config)
    _maybe_eject_low_hit_persistent_cache(metadata, workspace, batch_size, req_ids, past_lens)
    return out



def _make_async_cache_update_slot(backend: Any) -> dict[str, Any]:
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    workspace_buffer = torch.empty_like(backend.workspace_buffer)
    return {
        "workspace": workspace_buffer,
        "event": torch.cuda.Event(),
        "busy": False,
        "prefill_wrapper": BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            "NHD",
            backend=getattr(backend, "prefill_backend", "fa2"),
        ),
    }


def _ensure_async_cache_update_slots(backend: Any) -> None:
    deferred_count = int(getattr(backend, "mac_async_cache_update_deferred_count", 0) or 0)
    if deferred_count <= 0:
        return
    if _is_torch_compiling() or _is_cuda_graph_capturing() or _is_piecewise_cuda_graph_active():
        return
    slots = getattr(backend, "mac_async_cache_update_slots", None)
    if slots is None:
        slots = []
        backend.mac_async_cache_update_slots = slots
    for _ in range(deferred_count):
        slots.append(_make_async_cache_update_slot(backend))
    backend.mac_async_cache_update_deferred_count = 0


def _acquire_async_cache_update_slot(backend: Any) -> Optional[dict[str, Any]]:
    _ensure_async_cache_update_slots(backend)
    slots = getattr(backend, "mac_async_cache_update_slots", None)
    if not slots:
        return None
    for slot in slots:
        if not slot["busy"] or slot["event"].query():
            slot["busy"] = True
            return slot
    slot = slots[backend.mac_async_cache_update_next_slot]
    backend.mac_async_cache_update_next_slot = (backend.mac_async_cache_update_next_slot + 1) % len(slots)
    torch.cuda.current_stream().wait_event(slot["event"])
    slot["busy"] = True
    return slot


def _record_stream_recursive(stream: torch.cuda.Stream, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        value.record_stream(stream)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _record_stream_recursive(stream, item)
    elif isinstance(value, dict):
        for item in value.values():
            _record_stream_recursive(stream, item)


def _mark_cache_update_scheduled(mac_metadata: Any, slot: Optional[dict[str, Any]], stream: torch.cuda.Stream) -> None:
    mac_metadata.offload_event.record(stream)
    mac_metadata.offload_event_recorded = True
    mac_metadata.wait_cache_update_after_ffn = False
    if slot is not None:
        slot["event"].record(stream)


def _get_prefill_cache_lengths(mac_metadata: Any, forward_batch: Any):
    batch_size = mac_metadata.req_ids.numel()
    lens = (mac_metadata.offsets[1 : batch_size + 1] - mac_metadata.offsets[:batch_size]).contiguous()
    request_length = mac_metadata.req_len.clone()
    if forward_batch.extend_prefix_lens is not None:
        prefix_lens = forward_batch.extend_prefix_lens[:batch_size].to(
            device=mac_metadata.device, dtype=torch.int32, non_blocking=True
        )
    else:
        prefix_lens = forward_batch.seq_lens[:batch_size].to(
            device=mac_metadata.device, dtype=torch.int32, non_blocking=True
        ) - lens
    request_length[mac_metadata.req_ids.long()] = prefix_lens
    return request_length, lens


def _get_graph_safe_prefill_cache_lengths(mac_metadata: Any, forward_batch: Any):
    """Return stable cache-update tensors for CUDA graph capture/replay.

    CUDA graph nodes retain raw tensor data pointers. The eager path can use
    short-lived clones because kernels launch immediately, but captured custom
    kernels must never reference clone storage that Python may free after
    capture. These buffers live on MACMetadata for the lifetime of the layer.
    """

    batch_size = int(mac_metadata.req_ids.numel())
    req_ids_buf = getattr(mac_metadata, "prefill_update_req_ids_buf", None)
    offsets_buf = getattr(mac_metadata, "prefill_update_offsets_buf", None)
    lens_buf = getattr(mac_metadata, "prefill_update_lens_buf", None)
    prefix_lens_buf = getattr(mac_metadata, "prefill_update_prefix_lens_buf", None)
    if (
        req_ids_buf is None
        or offsets_buf is None
        or lens_buf is None
        or prefix_lens_buf is None
        or req_ids_buf.shape[0] < batch_size
        or offsets_buf.shape[0] < batch_size + 1
        or lens_buf.shape[0] < batch_size
        or prefix_lens_buf.shape[0] < batch_size
    ):
        raise RuntimeError(
            "MAC graph-safe prefill cache buffers are not initialized or too small: "
            f"batch={batch_size}"
        )

    req_ids = req_ids_buf[:batch_size]
    offsets = offsets_buf[: batch_size + 1]
    lens = lens_buf[:batch_size]
    prefix_lens = prefix_lens_buf[:batch_size]

    req_ids.copy_(mac_metadata.req_ids)
    offsets.copy_(mac_metadata.offsets[: batch_size + 1])
    torch.sub(offsets[1 : batch_size + 1], offsets[:batch_size], out=lens)

    # SGLang's piecewise CUDA graph replay feeds seq_lens as a graph input, but
    # extend_prefix_lens is not part of the compiled subgraph input list. Reading
    # it here can capture a warmup/dummy pointer and replay stale prefix lengths.
    # Derive the prefix from graph inputs instead.
    seq_lens_i32 = forward_batch.seq_lens[:batch_size].to(
        device=mac_metadata.device, dtype=torch.int32, non_blocking=True
    )
    torch.sub(seq_lens_i32, lens, out=prefix_lens)
    prefix_lens.clamp_(min=0)
    return prefix_lens, lens, offsets, req_ids


def _offload_fast_cache(
    mac_metadata: Any,
    forward_batch: Any,
    query: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    config: Any,
    request_length: Optional[torch.Tensor] = None,
    lens: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
    req_ids: Optional[torch.Tensor] = None,
) -> None:
    if req_ids is None:
        req_ids = mac_metadata.req_ids
    batch_size = req_ids.numel()
    if lens is None or request_length is None:
        lens = mac_metadata.cur_len[:batch_size]
        torch.sub(
            mac_metadata.offsets[1 : batch_size + 1],
            mac_metadata.offsets[:batch_size],
            out=lens,
        )
        request_length = mac_metadata.req_len
        if (
            not forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.extend_prefix_lens is not None
        ):
            prefix_lens = forward_batch.extend_prefix_lens[:batch_size].to(
                device=mac_metadata.device, dtype=torch.int32, non_blocking=True
            )
        else:
            prefix_lens = forward_batch.seq_lens[:batch_size].to(
                device=mac_metadata.device, dtype=torch.int32, non_blocking=True
            ) - lens
        request_length[req_ids.long()] = prefix_lens
    if offsets is None:
        offsets = mac_metadata.offsets
    if query.shape != output.shape:
        raise RuntimeError(
            "MAC prefill cache update shape mismatch: "
            f"src_q={tuple(query.shape)} src_attn={tuple(output.shape)} "
            f"src_lse={tuple(softmax_lse.shape)} "
            f"batch={batch_size} "
            f"mode={getattr(forward_batch, 'forward_mode', None)} "
            f"num_token_non_padded={getattr(forward_batch, 'num_token_non_padded_cpu', None)} "
            f"starts={getattr(forward_batch, 'mac_prefill_cache_update_starts', None)}"
        )
    mac_metadata.cache_ext.mac_prefill_update_cache(
        mac_metadata.unified_query_cache,
        mac_metadata.unified_attn_cache,
        mac_metadata.unified_lse_cache,
        request_length,
        req_ids,
        query,
        output,
        softmax_lse,
        offsets,
        lens,
        config.mac_lookback_tokens_left,
        mac_metadata.num_heads,
        mac_metadata.head_dim,
    )


def _launch_prefill_cache_update_async(
    backend: Any,
    forward_batch: Any,
    config: Any,
    mac_metadata: Any,
    q_input: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    layer: Any,
) -> None:
    if os.environ.get("MAC_DEBUG_DISABLE_PREFILL_CACHE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        _mac_prefill_trace("prefill_cache_update skip disabled", layer)
        return
    layer_id = getattr(layer, "layer_id", None)
    batch_size = mac_metadata.req_ids.numel()
    q_to_cache = mac_metadata.cur_q_before_rope
    if q_to_cache is None:
        _mac_prefill_trace(
            "prefill_cache_update skip q_to_cache_none "
            f"q_input={tuple(q_input.shape)} "
            f"starts={getattr(forward_batch, 'mac_prefill_cache_update_starts', None)}",
            layer,
        )
        return
    graph_safe = _is_cuda_graph_capturing() or _is_piecewise_cuda_graph_active() or _is_torch_compiling()
    suffix_plan = build_prefill_suffix_plan(forward_batch, batch_size, q_input.shape[0])
    if suffix_plan is not None and suffix_plan.total_len <= 0:
        _mac_prefill_trace(
            "prefill_cache_update skip empty_suffix "
            f"q_input={tuple(q_input.shape)} "
            f"starts={getattr(forward_batch, 'mac_prefill_cache_update_starts', None)}",
            layer,
        )
        return

    if graph_safe:
        request_length, lens, offsets, req_ids = _get_graph_safe_prefill_cache_lengths(
            mac_metadata, forward_batch
        )
    else:
        request_length, lens = _get_prefill_cache_lengths(mac_metadata, forward_batch)
        req_ids = mac_metadata.req_ids.clone()
        offsets = mac_metadata.offsets[: batch_size + 1].clone()
    needs_rectification_plan = int(config.mac_semantic_pos_ahead) > 0 and not graph_safe
    needs_suffix_kv_compaction = suffix_plan is not None
    if needs_rectification_plan or needs_suffix_kv_compaction:
        qo_indptr = backend.qo_indptr[0][: batch_size + 1].clone()
        kv_indptr = backend.kv_indptr[0][: batch_size + 1].clone()
        kv_indices = _build_prefill_kv_indices(backend, forward_batch, batch_size)
        kv_last_page_len = backend.kv_last_page_len[:batch_size].clone()
    else:
        qo_indptr = None
        kv_indptr = None
        kv_indices = None
        kv_last_page_len = None
    o_source = o
    lse_source = lse

    if (
        graph_safe
        and suffix_plan is None
        and batch_size == 1
        and q_to_cache.shape[0] < q_input.shape[0]
    ):
        tail_len = int(q_to_cache.shape[0])
        q_input = q_input[-tail_len:].contiguous()
        o_source = o[-tail_len:].contiguous()
        lse_source = lse[-tail_len:].contiguous()
        lens.fill_(tail_len)
        offsets[:1].zero_()
        offsets[1:2].fill_(tail_len)
        seq_lens_i32 = forward_batch.seq_lens[:1].to(
            device=mac_metadata.device, dtype=torch.int32, non_blocking=True
        )
        torch.sub(seq_lens_i32, lens[:1], out=request_length[:1])
        request_length[:1].clamp_(min=0)

    if suffix_plan is not None:
        assert kv_indices is not None
        assert kv_last_page_len is not None
        q_input = compact_by_segments(q_input, suffix_plan.segments)
        if bool(getattr(mac_metadata, "cur_q_before_rope_compacted", False)):
            if q_to_cache.shape[0] < suffix_plan.total_len:
                _invalidate_persistent_ready(mac_metadata, req_ids)
                return
            if q_to_cache.shape[0] > suffix_plan.total_len:
                q_to_cache = q_to_cache[-int(suffix_plan.total_len) :]
            q_to_cache = q_to_cache.contiguous()
        else:
            q_to_cache = compact_by_segments(q_to_cache, suffix_plan.segments)
        o_source = compact_by_segments(o, suffix_plan.segments)
        lse_source = compact_by_segments(lse, suffix_plan.segments)

        req_index = torch.tensor(
            suffix_plan.req_indices,
            dtype=torch.long,
            device=mac_metadata.device,
        )
        req_ids = req_ids.index_select(0, req_index).contiguous()
        request_length = mac_metadata.req_len.clone()
        request_length[req_ids.long()] = torch.tensor(
            suffix_plan.request_lengths,
            dtype=torch.int32,
            device=mac_metadata.device,
        )
        lens = torch.tensor(
            suffix_plan.lens, dtype=torch.int32, device=mac_metadata.device
        )
        offsets = torch.tensor(
            suffix_plan.offsets, dtype=torch.int32, device=mac_metadata.device
        )
        qo_indptr = offsets.clone()
        kv_indptr = torch.tensor(
            suffix_plan.kv_offsets, dtype=torch.int32, device=mac_metadata.device
        )
        kv_indices = compact_kv_indices_by_segments(
            kv_indices, suffix_plan.segments, batch_size=batch_size
        )
        kv_last_page_len = kv_last_page_len.index_select(0, req_index).contiguous()
    elif q_to_cache.shape[0] != q_input.shape[0]:
        # Piecewise CUDA graph pads the transformer layer to a static capture
        # length, then SGLang's unified_attention_with_output narrows q/k/v to
        # num_token_non_padded_cpu before calling the attention backend. MAC
        # preserves q before that backend-local narrowing, so trim the cached q
        # view to the exact rows that produced o/lse. This keeps the MAC cache
        # update in the graph while avoiding padded rows that FlashInfer never
        # computed attention for.
        if q_to_cache.shape[0] < q_input.shape[0]:
            raise RuntimeError(
                "MAC preserved fewer q rows than the attention backend received: "
                f"q_to_cache={tuple(q_to_cache.shape)} q_input={tuple(q_input.shape)} "
                f"batch={batch_size} "
                f"mode={getattr(forward_batch, 'forward_mode', None)} "
                f"num_token_non_padded={getattr(forward_batch, 'num_token_non_padded_cpu', None)}"
            )
        q_to_cache = q_to_cache[: q_input.shape[0]].contiguous()

    slot = _acquire_async_cache_update_slot(backend)
    if int(config.mac_semantic_pos_ahead) > 0 and slot is None and not graph_safe:
        _mac_prefill_trace(
            "prefill_cache_update skip no_async_slot "
            f"semantic_ahead={config.mac_semantic_pos_ahead}",
            layer,
        )
        _invalidate_persistent_ready(mac_metadata, req_ids)
        return
    stream = mac_metadata.stream
    sync_prefill_for_cuda_graph = bool(
        getattr(backend, "mac_sync_prefill_cache_update_when_cuda_graph", False)
    )
    if graph_safe or sync_prefill_for_cuda_graph:
        # CUDA graph capture needs the cache-update kernels and ready-state
        # writes to be captured on the active graph stream. The normal async
        # side stream is valuable in eager serving, but during piecewise CUDA
        # graph compile/capture/replay it can leave MAC's prefill cache write
        # outside the stream ordering that SGLang records for the graph. We use
        # the same active-stream path for regular prefill when decode CUDA graph
        # is enabled, because SGLang can immediately enter replay after the last
        # prefill chunk and must observe a fully ordered MAC cache/ready state.
        stream = None
    graph_prefill_wrapper = None
    if graph_safe and int(config.mac_semantic_pos_ahead) > 0:
        if bool(getattr(backend, "mac_graph_rectification_ready", False)):
            graph_prefill_wrapper = getattr(backend, "mac_graph_rectification_wrapper", None)
    if stream is None:
        cache_o, cache_lse = o_source, lse_source
        rect_window = int(config.mac_semantic_pos_ahead)
        if rect_window > 0:
            rect_window -= 1
        if int(config.mac_semantic_pos_ahead) > 0:
            if graph_safe:
                wrapper = graph_prefill_wrapper
            else:
                wrapper = slot["prefill_wrapper"] if slot is not None else None
            if wrapper is None:
                _mac_prefill_trace(
                    "prefill_cache_update skip no_rectification_wrapper "
                    f"semantic_ahead={config.mac_semantic_pos_ahead} "
                    f"graph_safe={graph_safe}",
                    layer,
                )
                _invalidate_persistent_ready(mac_metadata, req_ids)
                if slot is not None:
                    slot["busy"] = False
                return
            if not graph_safe:
                with profile_block(
                    "mac.prefill.rectification_begin_forward",
                    layer=layer_id,
                    forward_batch=forward_batch,
                    cuda=True,
                ):
                    wrapper.begin_forward(
                        qo_indptr,
                        kv_indptr,
                        kv_indices,
                        kv_last_page_len,
                        q_input.shape[1],
                        k_buffer.shape[1],
                        mac_metadata.head_dim,
                        1,
                        q_data_type=mac_metadata.dtype,
                        window_left=rect_window,
                    )
            with profile_block(
                "mac.prefill.rectification_forward_lse",
                layer=layer_id,
                forward_batch=forward_batch,
                cuda=True,
            ):
                cache_o, cache_lse = wrapper.forward_return_lse(
                    q_input,
                    (k_buffer, v_buffer),
                    causal=not layer.is_cross_attention,
                    sm_scale=layer.scaling,
                    window_left=rect_window,
                    logits_soft_cap=layer.logit_cap,
                    k_scale=getattr(layer, "k_scale_float", None),
                    v_scale=getattr(layer, "v_scale_float", None),
                )
            cache_lse = cache_lse * math.log(2.0)
            with profile_block(
                "mac.prefill.merge_update_then_downdate",
                layer=layer_id,
                forward_batch=forward_batch,
                cuda=True,
            ):
                mac_metadata.merge_ext.mac_update_then_downdate_bf16_inplace(
                    o_source, lse_source, cache_o, cache_lse
                )
        with profile_block(
            "mac.prefill.cache_write",
            layer=layer_id,
            forward_batch=forward_batch,
            cuda=True,
        ):
            _offload_fast_cache(
                mac_metadata,
                forward_batch,
                q_to_cache,
                cache_o,
                cache_lse,
                config,
                request_length=request_length,
                lens=lens,
                offsets=offsets,
                req_ids=req_ids,
            )
            _mark_persistent_ready(mac_metadata, req_ids)
            if _mac_prefill_trace_enabled(layer):
                _mac_prefill_trace(
                    "prefill_cache_update marked_ready_sync "
                    f"batch={batch_size} "
                    f"q_cached={tuple(q_to_cache.shape)} "
                    f"lens={lens.detach().cpu().tolist() if torch.is_tensor(lens) else lens}",
                    layer,
                )
            if slot is not None:
                slot["busy"] = False
        return

    current_stream = torch.cuda.current_stream(mac_metadata.device)
    stream.wait_stream(current_stream)
    with torch.cuda.stream(stream):
        cache_o, cache_lse = o_source, lse_source
        rect_window = int(config.mac_semantic_pos_ahead)
        if rect_window > 0:
            rect_window -= 1
        if config.mac_semantic_pos_ahead > 0 and slot is not None:
            wrapper = slot["prefill_wrapper"]
            with profile_block(
                "mac.prefill.rectification_begin_forward",
                layer=layer_id,
                forward_batch=forward_batch,
                cuda=True,
            ):
                wrapper.begin_forward(
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len,
                    q_input.shape[1],
                    k_buffer.shape[1],
                    mac_metadata.head_dim,
                    1,
                    q_data_type=mac_metadata.dtype,
                    window_left=rect_window,
                )
            with profile_block(
                "mac.prefill.rectification_forward_lse",
                layer=layer_id,
                forward_batch=forward_batch,
                cuda=True,
            ):
                cache_o, cache_lse = wrapper.forward_return_lse(
                    q_input,
                    (k_buffer, v_buffer),
                    causal=not layer.is_cross_attention,
                    sm_scale=layer.scaling,
                    window_left=rect_window,
                    logits_soft_cap=layer.logit_cap,
                    k_scale=getattr(layer, "k_scale_float", None),
                    v_scale=getattr(layer, "v_scale_float", None),
                )
            cache_lse = cache_lse * math.log(2.0)
            with profile_block(
                "mac.prefill.merge_update_then_downdate",
                layer=layer_id,
                forward_batch=forward_batch,
                cuda=True,
            ):
                mac_metadata.merge_ext.mac_update_then_downdate_bf16_inplace(
                    o_source, lse_source, cache_o, cache_lse
                )

        with profile_block(
            "mac.prefill.cache_write",
            layer=layer_id,
            forward_batch=forward_batch,
            cuda=True,
        ):
            _offload_fast_cache(
                mac_metadata,
                forward_batch,
                q_to_cache,
                cache_o,
                cache_lse,
                config,
                request_length=request_length,
                lens=lens,
                offsets=offsets,
                req_ids=req_ids,
            )
        _mark_persistent_ready(mac_metadata, req_ids)
        _mac_prefill_trace(
            "prefill_cache_update marked_ready_async "
            f"batch={batch_size} "
            f"q_cached={tuple(q_to_cache.shape)} "
            f"lens={lens.detach().cpu().tolist() if torch.is_tensor(lens) else lens}",
            layer,
        )
        _mark_cache_update_scheduled(mac_metadata, slot, stream)

    _record_stream_recursive(
        stream,
        (
            q_input,
            k_buffer,
            v_buffer,
            cache_o,
            cache_lse,
            o_source,
            lse_source,
            q_to_cache,
            mac_metadata.unified_query_cache,
            mac_metadata.unified_attn_cache,
            mac_metadata.unified_lse_cache,
            req_ids,
            offsets,
            request_length,
            lens,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            forward_batch.seq_lens,
        ),
    )


def flashinfer_forward_decode_around(
    original_fn: Any,
    self: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: Any,
    forward_batch: Any,
    save_kv_cache: bool = True,
    **kwargs: Any,
) -> Any:
    layer_id = getattr(layer, "layer_id", None)
    graph_capture = _is_cuda_graph_capturing()
    if graph_capture:
        _graph_trace(
            "decode enter "
            f"q={tuple(q.shape) if torch.is_tensor(q) else None} "
            f"k={tuple(k.shape) if torch.is_tensor(k) else None} "
            f"save_kv_cache={save_kv_cache}",
            layer,
        )
    if not _supports_mac_backend(self, layer, forward_batch):
        if graph_capture:
            _record_decode_graph_capture(self, forward_batch, persistent=False)
            _graph_trace("decode unsupported -> original", layer)
        if not profiling_enabled():
            return original_fn(
                self, q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )
        with profile_block(
            "flashinfer.forward_decode.original",
            layer=layer_id,
            forward_batch=forward_batch,
            cuda=True,
        ):
            return original_fn(
                self, q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )

    config = getattr(layer, "mac_config", None)
    if config is None:
        config = getattr(self, "mac_config", None)
    if config is None:
        config = get_mac_config()
    mac_metadata = layer.mac_metadata
    cache_loc = (
        forward_batch.out_cache_loc
        if not layer.is_cross_attention
        else forward_batch.encoder_out_cache_loc
    )
    if os.environ.get("MAC_DEBUG_BYPASS_DECODE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        if graph_capture:
            _record_decode_graph_capture(self, forward_batch, persistent=False)
            _graph_trace("decode bypass -> original", layer)
        return original_fn(
            self, q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )
    if not profiling_enabled():
        if k is not None:
            assert v is not None
            if save_kv_cache:
                if graph_capture:
                    _graph_trace("decode before set_kv_buffer", layer)
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )
                if graph_capture:
                    _graph_trace("decode after set_kv_buffer", layer)

        q_input = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        if graph_capture:
            _graph_trace("decode before wait_cache_update", layer)
        mac_metadata.wait_for_cache_update()
        if graph_capture:
            _graph_trace("decode before persistent", layer)
        persistent_out = _try_persistent_decode(
            self,
            forward_batch,
            config,
            mac_metadata,
            q_input,
            k_buffer,
            v_buffer,
            cache_loc,
            layer,
        )
        if graph_capture:
            _graph_trace(f"decode after persistent hit={persistent_out is not None}", layer)
        if persistent_out is not None:
            _record_decode_graph_capture(self, forward_batch, persistent=True)
            return persistent_out.view(-1, layer.tp_q_head_num * layer.head_dim)
        if graph_capture:
            _record_decode_graph_capture(self, forward_batch, persistent=False)
        _invalidate_persistent_ready(mac_metadata, mac_metadata.req_ids[: int(forward_batch.batch_size)])
        if graph_capture:
            _graph_trace("decode fallback -> original", layer)
        return original_fn(
            self, q, k, v, layer, forward_batch, save_kv_cache=False, **kwargs
        )

    if k is not None:
        assert v is not None
        if save_kv_cache:
            with profile_block(
                "sglang.kv_cache.set_decode",
                layer=layer_id,
                forward_batch=forward_batch,
                cuda=True,
            ):
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

    q_input = q.view(-1, layer.tp_q_head_num, layer.head_dim)
    with profile_block(
        "sglang.kv_cache.get_buffer",
        layer=layer_id,
        forward_batch=forward_batch,
        cuda=False,
    ):
        k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    with profile_block(
        "mac.cache.wait_for_update",
        layer=layer_id,
        forward_batch=forward_batch,
        cuda=True,
    ):
        mac_metadata.wait_for_cache_update()
    with profile_block(
        "mac.decode.persistent",
        layer=layer_id,
        forward_batch=forward_batch,
        cuda=True,
    ):
        persistent_out = _try_persistent_decode(
            self,
            forward_batch,
            config,
            mac_metadata,
            q_input,
            k_buffer,
            v_buffer,
            cache_loc,
            layer,
    )
    if persistent_out is not None:
        _record_decode_graph_capture(self, forward_batch, persistent=True)
        return persistent_out.view(-1, layer.tp_q_head_num * layer.head_dim)
    if graph_capture:
        _record_decode_graph_capture(self, forward_batch, persistent=False)
    _invalidate_persistent_ready(mac_metadata, mac_metadata.req_ids[: int(forward_batch.batch_size)])
    with profile_block(
        "flashinfer.forward_decode.original.persistent_fallback",
        layer=layer_id,
        forward_batch=forward_batch,
        cuda=True,
    ):
        return original_fn(
            self, q, k, v, layer, forward_batch, save_kv_cache=False, **kwargs
        )


def _make_flashinfer_forward_decode_direct(original_fn: Any) -> Any:
    def forward_decode(
        self: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Any,
        forward_batch: Any,
        save_kv_cache: bool = True,
        **kwargs: Any,
    ) -> Any:
        return flashinfer_forward_decode_around(
            original_fn,
            self,
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )

    return forward_decode


flashinfer_forward_decode_around.__mac_attention_direct_factory__ = (
    _make_flashinfer_forward_decode_direct
)


def flashinfer_forward_extend_around(
    original_fn: Any,
    self: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: Any,
    forward_batch: Any,
    save_kv_cache: bool = True,
    **kwargs: Any,
) -> Any:
    layer_id = getattr(layer, "layer_id", None)
    if not _supports_mac_backend(self, layer, forward_batch):
        if not profiling_enabled():
            return original_fn(
                self, q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )
        with profile_block(
            "flashinfer.forward_extend.original",
            layer=layer_id,
            forward_batch=forward_batch,
            cuda=True,
        ):
            return original_fn(
                self, q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )
    if getattr(self.forward_metadata, "use_ragged", True):
        if not profiling_enabled():
            return original_fn(
                self, q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )
        with profile_block(
            "flashinfer.forward_extend.original.ragged",
            layer=layer_id,
            forward_batch=forward_batch,
            cuda=True,
        ):
            return original_fn(
                self, q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )

    config = getattr(layer, "mac_config", None)
    if config is None:
        config = getattr(self, "mac_config", None)
    if config is None:
        config = get_mac_config()
    mac_metadata = layer.mac_metadata
    prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[self._get_wrapper_idx(layer)]
    cache_loc = (
        forward_batch.out_cache_loc
        if not layer.is_cross_attention
        else forward_batch.encoder_out_cache_loc
    )
    q = q.contiguous()
    q_input = q.view(-1, layer.tp_q_head_num, layer.head_dim)

    if not profiling_enabled():
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )
        k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        mac_metadata.wait_for_cache_update()

        from sglang.srt.layers.radix_attention import AttentionType

        causal = not layer.is_cross_attention and layer.attn_type != AttentionType.ENCODER_ONLY
        if _prefill_cache_update_required(
            forward_batch, mac_metadata.req_ids.numel(), q_input.shape[0]
        ):
            o, lse = prefill_wrapper_paged.forward_return_lse(
                q_input,
                (k_buffer, v_buffer),
                causal=causal,
                sm_scale=layer.scaling,
                window_left=layer.sliding_window_size,
                logits_soft_cap=layer.logit_cap,
                k_scale=getattr(layer, "k_scale_float", None),
                v_scale=getattr(layer, "v_scale_float", None),
            )
            lse = lse * math.log(2.0)
            _launch_prefill_cache_update_async(
                self,
                forward_batch,
                config,
                mac_metadata,
                q_input,
                k_buffer,
                v_buffer,
                o,
                lse,
                layer,
            )
        else:
            o = prefill_wrapper_paged.forward(
                q_input,
                (k_buffer, v_buffer),
                causal=causal,
                sm_scale=layer.scaling,
                window_left=layer.sliding_window_size,
                logits_soft_cap=layer.logit_cap,
                k_scale=getattr(layer, "k_scale_float", None),
                v_scale=getattr(layer, "v_scale_float", None),
            )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    if k is not None:
        assert v is not None
        if save_kv_cache:
            with profile_block(
                "sglang.kv_cache.set_extend",
                layer=layer_id,
                forward_batch=forward_batch,
                cuda=True,
            ):
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )
    with profile_block(
        "sglang.kv_cache.get_buffer",
        layer=layer_id,
        forward_batch=forward_batch,
        cuda=False,
    ):
        k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    with profile_block(
        "mac.cache.wait_for_update",
        layer=layer_id,
        forward_batch=forward_batch,
        cuda=True,
    ):
        mac_metadata.wait_for_cache_update()

    from sglang.srt.layers.radix_attention import AttentionType

    causal = not layer.is_cross_attention and layer.attn_type != AttentionType.ENCODER_ONLY
    if _prefill_cache_update_required(
        forward_batch, mac_metadata.req_ids.numel(), q_input.shape[0]
    ):
        with profile_block(
            "mac.prefill.flashinfer_paged_forward_lse",
            layer=layer_id,
            forward_batch=forward_batch,
            cuda=True,
        ):
            o, lse = prefill_wrapper_paged.forward_return_lse(
                q_input,
                (k_buffer, v_buffer),
                causal=causal,
                sm_scale=layer.scaling,
                window_left=layer.sliding_window_size,
                logits_soft_cap=layer.logit_cap,
                k_scale=getattr(layer, "k_scale_float", None),
                v_scale=getattr(layer, "v_scale_float", None),
            )
        lse = lse * math.log(2.0)
        with profile_block(
            "mac.prefill.cache_update_schedule",
            layer=layer_id,
            forward_batch=forward_batch,
            cuda=True,
        ):
            _launch_prefill_cache_update_async(
                self,
                forward_batch,
                config,
                mac_metadata,
                q_input,
                k_buffer,
                v_buffer,
                o,
                lse,
                layer,
            )
    else:
        with profile_block(
            "mac.prefill.flashinfer_paged_forward_no_lse",
            layer=layer_id,
            forward_batch=forward_batch,
            cuda=True,
        ):
            o = prefill_wrapper_paged.forward(
                q_input,
                (k_buffer, v_buffer),
                causal=causal,
                sm_scale=layer.scaling,
                window_left=layer.sliding_window_size,
                logits_soft_cap=layer.logit_cap,
                k_scale=getattr(layer, "k_scale_float", None),
                v_scale=getattr(layer, "v_scale_float", None),
            )
    return o.view(-1, layer.tp_q_head_num * layer.head_dim)


def _make_flashinfer_forward_extend_direct(original_fn: Any) -> Any:
    def forward_extend(
        self: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Any,
        forward_batch: Any,
        save_kv_cache: bool = True,
        **kwargs: Any,
    ) -> Any:
        return flashinfer_forward_extend_around(
            original_fn,
            self,
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )

    return forward_extend


flashinfer_forward_extend_around.__mac_attention_direct_factory__ = (
    _make_flashinfer_forward_extend_direct
)
