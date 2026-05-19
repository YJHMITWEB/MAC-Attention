from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch

from .bridge import (
    load_mac_decode_rope_preserve_extension,
    load_mac_merge_downdate_cache_extension,
    load_mac_persistent_decode_extension,
    load_mac_prefill_update_cache_extension,
)
from .config import get_mac_config
from .prefill_suffix import build_prefill_suffix_plan, compact_by_segments
from .profiling import enabled as profiling_enabled
from .profiling import mode_name, profile_block


logger = logging.getLogger(__name__)


def _graph_trace(message: str, layer_idx: int | None = None) -> None:
    if os.environ.get("MAC_DEBUG_GRAPH_TRACE", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    prefix = "[mac-graph-trace]"
    if layer_idx is not None:
        prefix += f"[layer={layer_idx}]"
    print(f"{prefix} {message}", file=sys.stderr, flush=True)


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _profile_logger(log_file: str) -> logging.Logger:
    rank = _rank()
    prof_logger = logging.getLogger(f"mac_attention_sglang_worker{rank}")
    prof_logger.setLevel(logging.INFO)
    prof_logger.propagate = False
    if not prof_logger.handlers:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        prof_logger.addHandler(fh)
    return prof_logger


def _mean_median(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = 0.5 * (ordered[mid - 1] + ordered[mid])
    return float(sum(values) / len(values)), float(median)


def _mac_use_fused_kv_rope() -> bool:
    return os.environ.get("MAC_USE_FUSED_KV_ROPE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _mac_use_fused_q_preserve_rope() -> bool:
    return os.environ.get("MAC_USE_FUSED_Q_PRESERVE_ROPE", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _try_mac_decode_rope_preserve(
    attn_module: Any,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    forward_batch: Any,
    metadata: Any,
) -> bool:
    if not _mac_use_fused_q_preserve_rope():
        return False
    if not forward_batch.forward_mode.is_decode_or_idle():
        return False
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return False
    if q.dim() != 2 or k.dim() != 2 or v.dim() != 2:
        return False
    if q.shape[0] != int(forward_batch.batch_size):
        return False
    if not hasattr(attn_module.rotary_emb, "cos_sin_cache"):
        return False
    cos_sin_cache = attn_module.rotary_emb.cos_sin_cache
    if not torch.is_tensor(cos_sin_cache) or cos_sin_cache.dtype != torch.float32:
        return False
    q_pre_buf = getattr(metadata, "decode_q_before_rope_buf", None)
    if (
        q_pre_buf is None
        or q_pre_buf.device != q.device
        or q_pre_buf.dtype != q.dtype
        or q_pre_buf.shape[0] < q.shape[0]
    ):
        return False
    try:
        k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
            attn_module.attn.layer_id
        )
    except Exception:
        return False
    if k_buffer.dtype != torch.bfloat16 or v_buffer.dtype != torch.bfloat16:
        return False

    q_pre = q_pre_buf[: q.shape[0]]
    metadata.cur_q_before_rope = q_pre
    metadata.cur_q_before_rope_local_start = 0
    metadata.cur_q_before_rope_local_len = int(q.shape[0])
    metadata.cur_q_before_rope_compacted = False
    ext = load_mac_decode_rope_preserve_extension(
        verbose=os.environ.get("MAC_EXTENSION_VERBOSE", "0") == "1"
    )
    ext.mac_decode_rope_preserve_q_bf16(
        q,
        k,
        v,
        q_pre,
        k_buffer,
        v_buffer,
        cos_sin_cache,
        positions,
        forward_batch.out_cache_loc,
        int(attn_module.head_dim),
        bool(getattr(attn_module.rotary_emb, "is_neox_style", True)),
    )
    return True


def _maybe_fused_kv_rope_arg(
    attn_module: Any,
    v: torch.Tensor,
    forward_batch: Any,
) -> Any:
    if not _mac_use_fused_kv_rope():
        return None
    try:
        from sglang.srt.models.utils import (
            create_fused_set_kv_buffer_arg,
            enable_fused_set_kv_buffer,
        )

        if not enable_fused_set_kv_buffer(forward_batch):
            return None
        return create_fused_set_kv_buffer_arg(
            value=v,
            layer=attn_module.attn,
            forward_batch=forward_batch,
        )
    except Exception:
        return None


def _cpu_int_list(value: Any, batch_size: int) -> Optional[list[int]]:
    if _is_torch_compiling() or _is_piecewise_cuda_graph_active():
        return None
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


def _allow_extension_methods_in_graph(module: Any, names: tuple[str, ...]) -> None:
    for name in names:
        fn = getattr(module, name, None)
        if fn is None:
            continue
        try:
            setattr(module, name, torch.compiler.allow_in_graph(fn))
        except Exception:
            # Older PyTorch builds may not expose allow_in_graph for pybind
            # callables. In that case the eager path still works; CUDA-graph
            # compile will report the unsupported callable directly.
            pass


def _req_pool_indices_host(forward_batch: Any, batch_size: int) -> Optional[list[int]]:
    if _is_torch_compiling() or _is_piecewise_cuda_graph_active():
        return None
    req_ids_src = getattr(forward_batch, "req_pool_indices", None)
    if req_ids_src is None:
        return None
    req_ids_src = req_ids_src[:batch_size]
    if torch.is_tensor(req_ids_src) and req_ids_src.device.type != "cpu":
        if _is_cuda_graph_capturing():
            return None
        req_ids_src = req_ids_src.detach().cpu()
    try:
        return [int(x) for x in req_ids_src.tolist()]
    except Exception:
        return None


def _active_request_capacity(
    forward_batch: Any,
    max_running: int,
    batch_size: Optional[int] = None,
    req_ids_host: Optional[list[int]] = None,
) -> int:
    if batch_size is None:
        batch_size = int(getattr(forward_batch, "batch_size", 0))
    req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)
    req_to_token = getattr(req_to_token_pool, "req_to_token", None)
    # SGLang request-pool indices are not guaranteed to be zero-based inside a
    # CUDA graph replay; graph warmups and server warmups can occupy/release
    # request-pool rows before the real request arrives. During TorchDynamo
    # compile we avoid device->host req-id copies, so keep a small reserve even
    # when req_ids_host is unavailable.
    request_pool_pad = max(int(os.environ.get("MAC_REQUEST_POOL_PAD", "8")), 1)
    capacity = max(
        int(max_running) + request_pool_pad,
        int(batch_size) + request_pool_pad,
        1,
    )
    if torch.is_tensor(req_to_token) and req_to_token.dim() >= 1:
        # Match SGLang's actual request-pool address space, including its
        # padding row 0 used by CUDA graph capture.
        capacity = max(capacity, int(req_to_token.shape[0]))
    pool_size = getattr(req_to_token_pool, "size", None)
    if pool_size is not None:
        try:
            capacity = max(capacity, int(pool_size))
        except Exception:
            pass
    if req_ids_host is None:
        shared = getattr(forward_batch, "_mac_forward_batch_state", None)
        req_ids_host = getattr(shared, "req_ids_host", None)
    if req_ids_host is None:
        req_ids_host = _req_pool_indices_host(forward_batch, int(batch_size))
    if req_ids_host:
        max_req = max((int(req) for req in req_ids_host if int(req) >= 0), default=-1)
        capacity = max(capacity, max_req + 1)
    return capacity


class MACForwardBatchState:
    def __init__(self) -> None:
        self.device: Optional[torch.device] = None
        self.max_running = 0
        self.batch_size = 0
        self.is_decode = False
        self.any_eligible: Optional[bool] = None
        self.all_eligible: Optional[bool] = None
        self.max_seq_len_cpu: Optional[int] = None

        self.req_ids_buf: Optional[torch.Tensor] = None
        self.req_ids: Optional[torch.Tensor] = None
        self.req_ids_long: Optional[torch.Tensor] = None
        self.req_ids_host: Optional[list[int]] = None
        self.seq_lens_host: Optional[list[int]] = None
        self.req_len: Optional[torch.Tensor] = None
        self.cur_len: Optional[torch.Tensor] = None
        self.cache_lens: Optional[torch.Tensor] = None
        self.match_req_len: Optional[torch.Tensor] = None
        self.offsets: Optional[torch.Tensor] = None

    def ensure(self, device: torch.device, max_running: int) -> None:
        needs_alloc = (
            self.req_ids_buf is None
            or self.device != device
            or self.max_running < max_running
        )
        if not needs_alloc:
            return
        self.device = device
        self.max_running = int(max_running)
        self.req_ids_buf = torch.empty((max_running,), dtype=torch.int32, device=device)
        self.req_len = torch.empty((max_running,), dtype=torch.int32, device=device)
        self.cur_len = torch.empty((max_running,), dtype=torch.int32, device=device)
        self.cache_lens = torch.empty((max_running,), dtype=torch.int32, device=device)
        self.match_req_len = torch.empty((max_running,), dtype=torch.int32, device=device)
        self.offsets = torch.zeros((max_running + 1,), dtype=torch.int32, device=device)

    def prepare(self, forward_batch: Any, device: torch.device, max_running: int, config: Any) -> None:
        batch_size = int(getattr(forward_batch, "batch_size", 0))
        req_ids_host = _req_pool_indices_host(forward_batch, batch_size)
        request_capacity = _active_request_capacity(
            forward_batch,
            max_running,
            batch_size,
            req_ids_host,
        )
        self.ensure(device, request_capacity)
        assert self.req_ids_buf is not None
        assert self.req_len is not None
        assert self.cur_len is not None
        assert self.cache_lens is not None
        assert self.match_req_len is not None
        assert self.offsets is not None

        self.batch_size = batch_size
        self.is_decode = bool(forward_batch.forward_mode.is_decode_or_idle())
        self.any_eligible = None
        self.all_eligible = None
        self.max_seq_len_cpu = None

        self.req_ids = self.req_ids_buf[:batch_size]
        self.req_ids_host = None
        self.seq_lens_host = None
        if batch_size == 0:
            self.req_ids_long = None
            return

        req_ids = forward_batch.req_pool_indices[:batch_size].to(
            device=device, dtype=torch.int32, non_blocking=True
        )
        self.req_ids.copy_(req_ids)
        self.req_ids_long = self.req_ids.long()
        self.req_ids_host = req_ids_host

        if self.is_decode:
            self.cur_len[:batch_size].fill_(1)
        elif forward_batch.extend_seq_lens is not None:
            self.cur_len[:batch_size].copy_(
                forward_batch.extend_seq_lens[:batch_size].to(
                    device=device, dtype=torch.int32, non_blocking=True
                )
            )
        elif forward_batch.extend_seq_lens_cpu is not None:
            self.cur_len[:batch_size].copy_(
                torch.tensor(
                    forward_batch.extend_seq_lens_cpu[:batch_size],
                    dtype=torch.int32,
                    device=device,
                )
            )
        else:
            self.cur_len[:batch_size].copy_(
                forward_batch.seq_lens[:batch_size].to(
                    device=device, dtype=torch.int32, non_blocking=True
                )
            )

        self.offsets[:1].zero_()
        torch.cumsum(self.cur_len[:batch_size], dim=0, out=self.offsets[1 : batch_size + 1])

        seq_lens_i32 = forward_batch.seq_lens[:batch_size].to(
            device=device, dtype=torch.int32, non_blocking=True
        )
        self.req_len[self.req_ids_long] = seq_lens_i32
        torch.clamp(seq_lens_i32 - self.cur_len[:batch_size], min=0, out=self.cache_lens[:batch_size])

        self.match_req_len.copy_(self.req_len)
        self.match_req_len[self.req_ids_long] = self.cache_lens[:batch_size]

        seq_lens_cpu = _cpu_int_list(getattr(forward_batch, "seq_lens_cpu", None), batch_size)
        if seq_lens_cpu is not None:
            self.seq_lens_host = seq_lens_cpu
            self.max_seq_len_cpu = max(seq_lens_cpu) if seq_lens_cpu else 0
            if self.is_decode:
                cur_lens_cpu = [1] * batch_size
            else:
                cur_lens_cpu = _cpu_int_list(getattr(forward_batch, "extend_seq_lens_cpu", None), batch_size)
            if cur_lens_cpu is not None:
                cache_lens_cpu = [max(s - c, 0) for s, c in zip(seq_lens_cpu, cur_lens_cpu)]
                eligible_cpu = [
                    c >= int(config.mac_gen_min_limit)
                    and int(config.mac_lookback_tokens_right) + 1 < c
                    for c in cache_lens_cpu
                ]
                self.any_eligible = any(eligible_cpu)
                self.all_eligible = all(eligible_cpu)


def _prepare_mac_forward_batch_state(
    forward_batch: Any,
    device: torch.device,
    config: Any,
    server_args: Any,
) -> None:
    if forward_batch is None or device.type != "cuda":
        return
    if _is_torch_compiling() or _is_piecewise_cuda_graph_active():
        return
    max_running = int(getattr(server_args, "max_running_requests", getattr(forward_batch, "batch_size", 1)))
    state = getattr(forward_batch, "_mac_forward_batch_state", None)
    if state is None:
        state = MACForwardBatchState()
        setattr(forward_batch, "_mac_forward_batch_state", state)
    state.prepare(forward_batch, device, max_running, config)


def _prefill_preserve_q_range(forward_batch: Any, q_len: int) -> tuple[int, int, bool]:
    batch_size = int(getattr(forward_batch, "batch_size", 0))
    plan = build_prefill_suffix_plan(
        forward_batch, batch_size, q_len, allow_backend_fallback=False
    )
    if plan is None:
        return 0, int(q_len), True
    if plan.total_len <= 0:
        return 0, 0, False
    if len(plan.segments) != 1 or plan.segments[0].batch_index != 0:
        return -1, int(plan.total_len), True
    segment = plan.segments[0]
    try:
        prefix_len = int(forward_batch.extend_prefix_lens_cpu[0])
    except Exception:
        return 0, int(q_len), True
    return max(0, segment.update_start - prefix_len), segment.length, True


def _preserve_q_before_rope(
    metadata: "MACMetadata",
    q: torch.Tensor,
    current_is_decode: bool,
    forward_batch: Any = None,
) -> None:
    q_before_rope_view = q.reshape(q.shape[0], metadata.num_heads, metadata.head_dim)
    metadata.cur_q_before_rope_local_start = 0
    metadata.cur_q_before_rope_local_len = int(q_before_rope_view.shape[0])
    metadata.cur_q_before_rope_compacted = False
    decode_q_buf = getattr(metadata, "decode_q_before_rope_buf", None)
    if (
        current_is_decode
        and decode_q_buf is not None
        and decode_q_buf.device == q.device
        and decode_q_buf.dtype == q.dtype
        and decode_q_buf.shape[0] >= q_before_rope_view.shape[0]
    ):
        metadata.cur_q_before_rope = decode_q_buf[: q_before_rope_view.shape[0]]
        metadata.cur_q_before_rope.copy_(q_before_rope_view)
        return

    if current_is_decode or forward_batch is None:
        metadata.cur_q_before_rope = q_before_rope_view.contiguous().clone()
        return

    if _is_torch_compiling() or _is_piecewise_cuda_graph_active():
        q_len = int(q_before_rope_view.shape[0])
        tail_len = min(max(1, int(getattr(metadata, "prefill_preserve_tail_tokens", 1))), q_len)
        local_start = q_len - tail_len
        prefill_q_buf = getattr(metadata, "prefill_q_before_rope_buf", None)
        if (
            prefill_q_buf is not None
            and prefill_q_buf.device == q.device
            and prefill_q_buf.dtype == q.dtype
            and prefill_q_buf.shape[0] >= tail_len
        ):
            prefill_q_buf[:tail_len].copy_(q_before_rope_view[local_start:q_len])
            metadata.cur_q_before_rope = prefill_q_buf[:tail_len]
            metadata.cur_q_before_rope_local_start = int(local_start)
            metadata.cur_q_before_rope_local_len = int(tail_len)
            metadata.cur_q_before_rope_compacted = True
            return

    plan = build_prefill_suffix_plan(
        forward_batch,
        int(getattr(forward_batch, "batch_size", 0)),
        q_before_rope_view.shape[0],
        allow_backend_fallback=False,
    )
    if plan is None:
        local_start, local_len, should_preserve = (
            0,
            int(q_before_rope_view.shape[0]),
            True,
        )
    elif plan.total_len <= 0:
        local_start, local_len, should_preserve = 0, 0, False
    elif len(plan.segments) == 1 and plan.segments[0].batch_index == 0:
        segment = plan.segments[0]
        try:
            prefix_len = int(forward_batch.extend_prefix_lens_cpu[0])
        except Exception:
            local_start = 0
        else:
            local_start = max(0, segment.update_start - prefix_len)
        local_len, should_preserve = segment.length, True
    else:
        local_start, local_len, should_preserve = -1, int(plan.total_len), True
    metadata.cur_q_before_rope_compacted = plan is not None
    metadata.cur_q_before_rope_local_start = int(local_start)
    metadata.cur_q_before_rope_local_len = int(local_len)
    if not should_preserve:
        metadata.cur_q_before_rope = None
        return
    prefill_q_buf = getattr(metadata, "prefill_q_before_rope_buf", None)
    if (
        prefill_q_buf is not None
        and prefill_q_buf.device == q.device
        and prefill_q_buf.dtype == q.dtype
        and prefill_q_buf.shape[0] >= q_before_rope_view.shape[0]
    ):
        # In SGLang's piecewise CUDA graph path, Python attribute writes inside
        # compiled regions are not replayed as dataflow. Keep a stable prefill
        # Q buffer and capture the preservation as a tensor mutation instead.
        prefill_q_buf[: q_before_rope_view.shape[0]].copy_(q_before_rope_view)
        metadata.cur_q_before_rope = prefill_q_buf
        metadata.cur_q_before_rope_compacted = False
        return
    if plan is not None:
        metadata.cur_q_before_rope = compact_by_segments(
            q_before_rope_view, plan.segments, clone=True
        )
        return
    if local_start == 0 and local_len == q_before_rope_view.shape[0]:
        metadata.cur_q_before_rope = q_before_rope_view.contiguous().clone()
        return
    metadata.cur_q_before_rope = q_before_rope_view[
        local_start : local_start + local_len
    ].contiguous().clone()


class MACMetadata:
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int) -> None:
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.batch_size = 0
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None
        self.layer_idx = 0

        self.stream: Optional[torch.cuda.Stream] = None
        self.offload_event: Optional[torch.cuda.Event] = None
        self.offload_event_recorded = False
        self.wait_cache_update_after_ffn = False
        self.pending_decode_cache_update = None
        self.last_forward_was_decode = False

        self.cur_q_before_rope: Optional[torch.Tensor] = None
        self.cur_q_before_rope_local_start = 0
        self.cur_q_before_rope_local_len = 0
        self.cur_q_before_rope_compacted = False
        self.decode_q_before_rope_buf: Optional[torch.Tensor] = None
        self.prefill_q_before_rope_buf: Optional[torch.Tensor] = None
        self.prefill_preserve_tail_tokens = 0
        self.prefill_update_req_ids_buf: Optional[torch.Tensor] = None
        self.prefill_update_offsets_buf: Optional[torch.Tensor] = None
        self.prefill_update_lens_buf: Optional[torch.Tensor] = None
        self.prefill_update_prefix_lens_buf: Optional[torch.Tensor] = None
        self.rotary_emb = None
        self.cur_q_len = 0
        self.cur_kv_len = 0
        self.num_k_head = 0

        self.unified_query_cache: Optional[torch.Tensor] = None
        self.unified_attn_cache: Optional[torch.Tensor] = None
        self.unified_lse_cache: Optional[torch.Tensor] = None
        self.hit: Optional[torch.Tensor] = None
        self.left: Optional[torch.Tensor] = None
        self.idx: Optional[torch.Tensor] = None
        self.cache_local_hit: Optional[torch.Tensor] = None
        self.cache_local_indices: Optional[torch.Tensor] = None
        self.req_len: Optional[torch.Tensor] = None
        self.cur_len: Optional[torch.Tensor] = None
        self.cache_lens: Optional[torch.Tensor] = None
        self.match_req_len: Optional[torch.Tensor] = None
        self.req_ids_long: Optional[torch.Tensor] = None
        self.req_ids_buf: Optional[torch.Tensor] = None
        self.req_ids: Optional[torch.Tensor] = None
        self.offsets: Optional[torch.Tensor] = None
        self.is_decode = False
        self.any_eligible: Optional[bool] = None
        self.all_eligible: Optional[bool] = None

        self.cache_ext = None
        self.merge_ext = None
        self.persistent_decode_ext = None

        self.request_epoch: Optional[torch.Tensor] = None
        self.mac_cache_ready: Optional[torch.Tensor] = None
        self.mac_cache_epoch: Optional[torch.Tensor] = None
        self.last_req_len: Optional[torch.Tensor] = None
        self.mac_cache_ready_host: Optional[list[bool]] = None
        self.last_req_len_host: Optional[list[int]] = None

        self.decoding_hit = 0
        self.decoding_miss = 0
        self.decoding_total_kv = 0
        self.decoding_skipped_kv = 0
        self.decoding_range: list[float] = []
        self.prefilling_hit = 0
        self.prefilling_miss = 0
        self.prefilling_total_kv = 0
        self.prefilling_skipped_kv = 0
        self.prefilling_range: list[float] = []
        self.continuous_hit = torch.zeros(self.num_heads)

    def _ensure_extensions(self) -> None:
        verbose = os.environ.get("MAC_EXTENSION_VERBOSE", "0") == "1"
        config = get_mac_config()
        if self.cache_ext is None:
            self.cache_ext = load_mac_prefill_update_cache_extension(verbose=verbose)
        if self.merge_ext is None:
            self.merge_ext = load_mac_merge_downdate_cache_extension(verbose=verbose)
        if self.persistent_decode_ext is None:
            self.persistent_decode_ext = load_mac_persistent_decode_extension(
                verbose=verbose,
                fast_math=bool(config.mac_persistent_fast_math),
            )

    def _ensure_stream(self, device: torch.device) -> None:
        if self.stream is None or self.device != device:
            self.stream = torch.cuda.Stream(
                device=device,
                priority=int(os.environ.get("MAC_CACHE_UPDATE_STREAM_PRIORITY", "1")),
            )
            self.offload_event = torch.cuda.Event()

    def init_host_buffer(
        self,
        dtype: torch.dtype,
        device: torch.device,
        num_query_token: int,
        batch_size: int,
        max_num_query: int,
        server_args: Any,
        layer_idx: int,
        config: Any = None,
        request_capacity: Optional[int] = None,
    ) -> None:
        if config is None:
            config = get_mac_config(server_args)
        self.device = device
        self.dtype = dtype
        self.layer_idx = int(layer_idx)
        self.cur_q_len = int(num_query_token)
        self.batch_size = int(batch_size)

        if not config.enable_mac:
            return
        if device.type != "cuda":
            return

        max_running = int(
            request_capacity
            if request_capacity is not None
            else getattr(server_args, "max_running_requests", batch_size)
        )
        capacity = int(config.mac_lookback_tokens_left)
        self.prefill_preserve_tail_tokens = max(1, capacity)
        prefill_capacity = max(int(max_num_query), int(num_query_token), 1)
        prefill_q_ready = (
            self.prefill_q_before_rope_buf is not None
            and self.prefill_q_before_rope_buf.device == device
            and self.prefill_q_before_rope_buf.dtype == dtype
            and self.prefill_q_before_rope_buf.shape[0] >= prefill_capacity
            and self.prefill_q_before_rope_buf.shape[1] == self.num_heads
            and self.prefill_q_before_rope_buf.shape[2] == self.head_dim
        )
        prefill_update_ready = (
            self.prefill_update_req_ids_buf is not None
            and self.prefill_update_req_ids_buf.device == device
            and self.prefill_update_req_ids_buf.shape[0] >= max_running
            and self.prefill_update_offsets_buf is not None
            and self.prefill_update_offsets_buf.device == device
            and self.prefill_update_offsets_buf.shape[0] >= max_running + 1
            and self.prefill_update_lens_buf is not None
            and self.prefill_update_lens_buf.device == device
            and self.prefill_update_lens_buf.shape[0] >= max_running
            and self.prefill_update_prefix_lens_buf is not None
            and self.prefill_update_prefix_lens_buf.device == device
            and self.prefill_update_prefix_lens_buf.shape[0] >= max_running
        )
        if (
            self.unified_query_cache is not None
            and self.unified_query_cache.device == device
            and self.unified_query_cache.dtype == dtype
            and self.unified_query_cache.shape[0] >= max_running
            and self.unified_query_cache.shape[1] == capacity
            and prefill_q_ready
            and prefill_update_ready
        ):
            if self.continuous_hit.device != device:
                self.continuous_hit = self.continuous_hit.to(device)
            return

        self._ensure_extensions()
        self._ensure_stream(device)

        needs_alloc = (
            self.unified_query_cache is None
            or self.unified_query_cache.device != device
            or self.unified_query_cache.dtype != dtype
            or self.unified_query_cache.shape[0] < max_running
            or self.unified_query_cache.shape[1] != capacity
        )
        if needs_alloc:
            self.hit = torch.empty((max_running, self.num_heads), device=device, dtype=torch.bool)
            self.left = torch.empty((max_running, self.num_heads), device=device, dtype=torch.int32)
            self.idx = torch.empty((max_running, self.num_heads), device=device, dtype=torch.int32)
            self.unified_query_cache = torch.empty(
                (max_running, capacity, self.num_heads, self.head_dim),
                dtype=dtype,
                device=device,
            )
            self.unified_attn_cache = torch.empty_like(self.unified_query_cache)
            self.unified_lse_cache = torch.empty(
                (max_running, capacity, self.num_heads),
                dtype=torch.float32,
                device=device,
            )
            self.decode_q_before_rope_buf = torch.empty(
                (max_running, self.num_heads, self.head_dim),
                dtype=dtype,
                device=device,
            )
            self.prefill_q_before_rope_buf = torch.empty(
                (prefill_capacity, self.num_heads, self.head_dim),
                dtype=dtype,
                device=device,
            )
            self.prefill_update_req_ids_buf = torch.empty(
                (max_running,), dtype=torch.int32, device=device
            )
            self.prefill_update_offsets_buf = torch.empty(
                (max_running + 1,), dtype=torch.int32, device=device
            )
            self.prefill_update_lens_buf = torch.empty(
                (max_running,), dtype=torch.int32, device=device
            )
            self.prefill_update_prefix_lens_buf = torch.empty(
                (max_running,), dtype=torch.int32, device=device
            )
            self.req_len = torch.empty((max_running,), dtype=torch.int32, device=device)
            self.cur_len = torch.empty((max_running,), dtype=torch.int32, device=device)
            self.cache_lens = torch.empty((max_running,), dtype=torch.int32, device=device)
            self.match_req_len = torch.empty((max_running,), dtype=torch.int32, device=device)
            self.req_ids_buf = torch.empty((max_running,), dtype=torch.int32, device=device)
            self.req_ids = self.req_ids_buf[:1]
            self.offsets = torch.zeros((max_running + 1,), dtype=torch.int32, device=device)
            self.request_epoch = torch.zeros((max_running,), dtype=torch.int64, device=device)
            self.mac_cache_ready = torch.zeros((max_running,), dtype=torch.bool, device=device)
            self.mac_cache_epoch = torch.full((max_running,), -1, dtype=torch.int64, device=device)
            self.last_req_len = torch.full((max_running,), -1, dtype=torch.int32, device=device)
            self.mac_cache_ready_host = [False] * max_running
            self.last_req_len_host = [-1] * max_running
        elif not prefill_q_ready:
            self.prefill_q_before_rope_buf = torch.empty(
                (prefill_capacity, self.num_heads, self.head_dim),
                dtype=dtype,
                device=device,
            )
        if not prefill_update_ready:
            self.prefill_update_req_ids_buf = torch.empty(
                (max_running,), dtype=torch.int32, device=device
            )
            self.prefill_update_offsets_buf = torch.empty(
                (max_running + 1,), dtype=torch.int32, device=device
            )
            self.prefill_update_lens_buf = torch.empty(
                (max_running,), dtype=torch.int32, device=device
            )
            self.prefill_update_prefix_lens_buf = torch.empty(
                (max_running,), dtype=torch.int32, device=device
            )
        if self.mac_cache_ready_host is None or len(self.mac_cache_ready_host) < max_running:
            old_ready = self.mac_cache_ready_host or []
            old_lens = self.last_req_len_host or []
            self.mac_cache_ready_host = old_ready + [False] * (max_running - len(old_ready))
            self.last_req_len_host = old_lens + [-1] * (max_running - len(old_lens))

        if self.continuous_hit.device != device:
            self.continuous_hit = self.continuous_hit.to(device)

    def _update_persistent_ready_host(
        self,
        req_ids_host: Optional[list[int]],
        seq_lens_host: Optional[list[int]],
    ) -> None:
        if self.mac_cache_ready_host is None or self.last_req_len_host is None:
            return
        if not req_ids_host:
            return
        invalidate_reqs: set[int] = set()
        if seq_lens_host is not None:
            for req, seq_len in zip(req_ids_host, seq_lens_host):
                if req < 0 or req >= len(self.mac_cache_ready_host):
                    continue
                old_len = self.last_req_len_host[req]
                if old_len >= 0 and int(seq_len) < old_len:
                    self.mac_cache_ready_host[req] = False
                    invalidate_reqs.add(int(req))
                self.last_req_len_host[req] = int(seq_len)
        if not self.is_decode:
            for req in req_ids_host:
                if 0 <= req < len(self.mac_cache_ready_host):
                    self.mac_cache_ready_host[req] = False
                    ejected = getattr(self, "mac_low_hit_ejected_host", None)
                    if ejected is not None and req < len(ejected):
                        ejected[req] = False
                    invalidate_reqs.add(int(req))
        if (
            invalidate_reqs
            and self.device is not None
            and self.mac_cache_ready is not None
            and self.mac_cache_epoch is not None
            and self.request_epoch is not None
        ):
            for req in sorted(invalidate_reqs):
                if req < 0 or req >= self.mac_cache_ready.shape[0]:
                    continue
                self.mac_cache_ready[req].fill_(False)
                self.mac_cache_epoch[req].fill_(-1)
                self.request_epoch[req].add_(1)

    def prepare_forward_batch(self, forward_batch: Any, config: Any = None) -> None:
        if self.device is None or self.req_ids_buf is None or self.req_len is None:
            return
        if config is None:
            config = get_mac_config()
        batch_size = int(forward_batch.batch_size)
        graph_safe = _is_torch_compiling() or _is_piecewise_cuda_graph_active()
        shared = None if graph_safe else getattr(forward_batch, "_mac_forward_batch_state", None)
        if (
            shared is not None
            and getattr(shared, "device", None) == self.device
            and getattr(shared, "req_ids", None) is not None
        ):
            self.batch_size = batch_size
            self.req_ids = shared.req_ids
            self.req_ids_long = shared.req_ids_long
            self.req_ids_host = shared.req_ids_host
            self.req_len = shared.req_len
            self.cur_len = shared.cur_len
            self.cache_lens = shared.cache_lens
            self.match_req_len = shared.match_req_len
            self.offsets = shared.offsets
            self.is_decode = shared.is_decode
            self.any_eligible = shared.any_eligible
            self.all_eligible = shared.all_eligible
            if shared.max_seq_len_cpu is not None:
                self.cur_kv_len = int(shared.max_seq_len_cpu)
            self._update_persistent_ready_host(
                shared.req_ids_host,
                shared.seq_lens_host,
            )
            return
        if batch_size == 0:
            return

        req_ids = forward_batch.req_pool_indices[:batch_size].to(
            device=self.device, dtype=torch.int32, non_blocking=True
        )
        self.req_ids_buf[:batch_size].copy_(req_ids)
        self.req_ids = self.req_ids_buf[:batch_size]
        self.req_ids_long = self.req_ids.long()
        if graph_safe:
            self.req_ids_host = None
        else:
            req_ids_src = forward_batch.req_pool_indices[:batch_size]
            if req_ids_src.device.type == "cpu":
                self.req_ids_host = [int(x) for x in req_ids_src.tolist()]
            else:
                self.req_ids_host = [int(x) for x in req_ids_src.detach().cpu().tolist()]
        self.is_decode = bool(forward_batch.forward_mode.is_decode_or_idle())
        self.any_eligible = None
        self.all_eligible = None

        if self.is_decode:
            self.cur_len[:batch_size].fill_(1)
        elif graph_safe:
            # PCG replay pads smaller requests into a fixed token graph. Avoid
            # Python CPU fields or dtype-unstable extend tensors here; the
            # static graph token extent is the stable value Dynamo captured.
            self.cur_len[:batch_size].fill_(int(forward_batch.input_ids.shape[0]))
        elif forward_batch.extend_seq_lens is not None:
            self.cur_len[:batch_size].copy_(
                forward_batch.extend_seq_lens[:batch_size].to(
                    device=self.device, dtype=torch.int32, non_blocking=True
                )
            )
        elif forward_batch.extend_seq_lens_cpu is not None:
            self.cur_len[:batch_size].copy_(
                torch.tensor(
                    forward_batch.extend_seq_lens_cpu[:batch_size],
                    dtype=torch.int32,
                    device=self.device,
                )
            )
        else:
            self.cur_len[:batch_size].copy_(
                forward_batch.seq_lens[:batch_size].to(
                    device=self.device, dtype=torch.int32, non_blocking=True
                )
            )

        self.offsets[:1].zero_()
        torch.cumsum(self.cur_len[:batch_size], dim=0, out=self.offsets[1 : batch_size + 1])
        seq_lens_i32 = forward_batch.seq_lens[:batch_size].to(
            device=self.device, dtype=torch.int32, non_blocking=True
        )
        seq_lens_host = None if graph_safe else _cpu_int_list(getattr(forward_batch, "seq_lens_cpu", None), batch_size)
        if seq_lens_host is not None and seq_lens_host:
            self.cur_kv_len = max(seq_lens_host)
        self._update_persistent_ready_host(self.req_ids_host, seq_lens_host)
        self.req_len[self.req_ids_long] = seq_lens_i32
        if self.cache_lens is not None:
            torch.clamp(seq_lens_i32 - self.cur_len[:batch_size], min=0, out=self.cache_lens[:batch_size])
        if self.match_req_len is not None:
            self.match_req_len.copy_(self.req_len)
            self.match_req_len[self.req_ids_long] = self.cache_lens[:batch_size]

    def wait_for_cache_update(self) -> None:
        if self.device is not None and self.offload_event_recorded and self.offload_event is not None:
            torch.cuda.current_stream(self.device).wait_event(self.offload_event)
            self.offload_event_recorded = False
        self.wait_cache_update_after_ffn = False

    def reset(self) -> None:
        self.wait_for_cache_update()
        self.pending_decode_cache_update = None
        self.cur_kv_len = 0
        self.last_forward_was_decode = False
        self.cache_local_hit = None
        self.cache_local_indices = None
        self.decoding_hit = 0
        self.decoding_miss = 0
        self.decoding_total_kv = 0
        self.decoding_skipped_kv = 0
        self.decoding_range = []
        self.prefilling_hit = 0
        self.prefilling_miss = 0
        self.prefilling_total_kv = 0
        self.prefilling_skipped_kv = 0
        self.prefilling_range = []
        if self.continuous_hit.device.type == "cuda":
            self.continuous_hit.zero_()
        else:
            self.continuous_hit = torch.zeros(self.num_heads)


def llama_attention_init_after(result: Any, self: Any, *args: Any, **kwargs: Any) -> Any:
    self.mac_metadata = MACMetadata(self.num_heads, self.num_kv_heads, self.head_dim)
    self.attn.mac_metadata = self.mac_metadata
    return result


def _log_profile_if_needed(metadata: MACMetadata, config: Any, layer_idx: int) -> None:
    if not config.mac_profile:
        return
    root = config.mac_profile_path or str(
        Path(os.environ.get("MAC_WORKSPACE_BASE", os.getcwd())) / "profiles" / "mac"
    )
    folder = (
        f"mac_chunkSize_unknown_genLimit{config.mac_gen_min_limit}_"
        f"threshold{config.mac_threshold}_left{config.mac_lookback_tokens_left}_"
        f"right{config.mac_lookback_tokens_right}_semAhead{config.mac_semantic_pos_ahead}"
    )
    prof_logger = _profile_logger(str(Path(root) / folder / f"{_rank()}.log"))
    pre_mean_med = _mean_median(metadata.prefilling_range)
    dec_mean_med = _mean_median(metadata.decoding_range)
    pre_hit_total = metadata.prefilling_hit + metadata.prefilling_miss
    dec_hit_total = metadata.decoding_hit + metadata.decoding_miss
    pre_hit_ratio = metadata.prefilling_hit / (pre_hit_total + 1e-8)
    dec_hit_ratio = metadata.decoding_hit / (dec_hit_total + 1e-8)
    pre_skip_ratio = metadata.prefilling_skipped_kv / (metadata.prefilling_total_kv + 1e-8)
    dec_skip_ratio = metadata.decoding_skipped_kv / (metadata.decoding_total_kv + 1e-8)
    continuous_hit = (
        metadata.continuous_hit.detach().cpu().tolist()
        if torch.is_tensor(metadata.continuous_hit)
        else []
    )
    prof_logger.info(
        (
            "layer %s prefilling ------- hit: %s, miss: %s, "
            "hit ratio: %.4f, overall skipped ratio: %.4f, "
            "avg/med fast range: %s tokens"
        ),
        layer_idx,
        metadata.prefilling_hit,
        metadata.prefilling_miss,
        pre_hit_ratio,
        pre_skip_ratio,
        pre_mean_med,
    )
    prof_logger.info(
        (
            "layer %s decoding ------- hit: %s, miss: %s, "
            "hit ratio: %.4f, overall skipped ratio: %.4f, "
            "avg/med fast range: %s tokens, max continuous hit per head: %s"
        ),
        layer_idx,
        metadata.decoding_hit,
        metadata.decoding_miss,
        dec_hit_ratio,
        dec_skip_ratio,
        dec_mean_med,
        continuous_hit,
    )


def llama_attention_forward_around(original_fn: Any, self: Any, *args: Any, **kwargs: Any) -> Any:
    positions = kwargs.get("positions", args[0] if len(args) > 0 else None)
    hidden_states = kwargs.get("hidden_states", args[1] if len(args) > 1 else None)
    forward_batch = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
    layer_idx = int(getattr(getattr(self, "attn", None), "layer_id", 0))
    if positions is None or hidden_states is None or forward_batch is None:
        with profile_block("llama.attention.forward.original", layer=layer_idx, cuda=True):
            return original_fn(self, *args, **kwargs)

    config = getattr(forward_batch, "_mac_config", None)
    if config is None:
        config = getattr(getattr(self, "attn", None), "mac_config", None)
    if config is None:
        config = get_mac_config()
    if not config.enable_mac or hidden_states.device.type != "cuda":
        with profile_block(
            "llama.attention.forward.original",
            layer=layer_idx,
            forward_batch=forward_batch,
            cuda=True,
        ):
            return original_fn(self, *args, **kwargs)

    try:
        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()
    except Exception:
        server_args = None

    metadata: MACMetadata = getattr(self, "mac_metadata", None)
    if metadata is None:
        metadata = MACMetadata(self.num_heads, self.num_kv_heads, self.head_dim)
        self.mac_metadata = metadata

    if not profiling_enabled():
        graph_capture = _is_cuda_graph_capturing()
        if graph_capture:
            _graph_trace(
                "attention enter "
                f"mode={getattr(forward_batch, 'forward_mode', None)} "
                f"hidden={tuple(hidden_states.shape)} "
                f"batch={getattr(forward_batch, 'batch_size', None)}",
                layer_idx,
            )
        capacity = int(config.mac_lookback_tokens_left)
        max_running = int(
            getattr(server_args, "max_running_requests", forward_batch.batch_size)
        )
        request_capacity = _active_request_capacity(forward_batch, max_running)
        cache_ready = (
            metadata.unified_query_cache is not None
            and metadata.unified_query_cache.device == hidden_states.device
            and metadata.unified_query_cache.dtype == hidden_states.dtype
            and metadata.unified_query_cache.shape[0] >= request_capacity
            and metadata.unified_query_cache.shape[1] == capacity
        )
        if cache_ready:
            metadata.device = hidden_states.device
            metadata.dtype = hidden_states.dtype
            metadata.layer_idx = layer_idx
            metadata.cur_q_len = int(hidden_states.shape[0])
            metadata.batch_size = int(forward_batch.batch_size)
        else:
            metadata.init_host_buffer(
                hidden_states.dtype,
                hidden_states.device,
                hidden_states.shape[0],
                forward_batch.batch_size,
                getattr(server_args, "chunked_prefill_size", hidden_states.shape[0]),
                server_args,
                layer_idx,
                config,
                request_capacity,
            )
        if graph_capture:
            _graph_trace("after init_host_buffer", layer_idx)
        metadata.prepare_forward_batch(forward_batch, config)
        if graph_capture:
            _graph_trace(
                "after prepare_forward_batch "
                f"is_decode={forward_batch.forward_mode.is_decode_or_idle()} "
                f"cur_kv_len={getattr(metadata, 'cur_kv_len', None)}",
                layer_idx,
            )

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if graph_capture:
            _graph_trace("after qkv_proj", layer_idx)
        current_is_decode = forward_batch.forward_mode.is_decode_or_idle()
        fused_q_rope = _try_mac_decode_rope_preserve(
            self, positions, q, k, v, forward_batch, metadata
        )
        if not fused_q_rope:
            _preserve_q_before_rope(metadata, q, current_is_decode, forward_batch)
        if graph_capture:
            _graph_trace(f"after preserve_q fused_q_rope={fused_q_rope}", layer_idx)

        fused_kv_arg = None
        if not fused_q_rope:
            fused_kv_arg = _maybe_fused_kv_rope_arg(self, v, forward_batch)
            q, k = self.rotary_emb(
                positions,
                q,
                k,
                fused_set_kv_buffer_arg=fused_kv_arg,
            )
        if graph_capture:
            _graph_trace(f"after rotary fused_kv={fused_kv_arg is not None}", layer_idx)
        metadata.rotary_emb = self.rotary_emb

        if metadata.last_forward_was_decode and not current_is_decode:
            _log_profile_if_needed(metadata, config, layer_idx)
            metadata.reset()

        metadata.num_k_head = getattr(self, "num_kv_heads", 0)
        self.attn.mac_metadata = metadata
        self.attn.mac_config = config
        if graph_capture:
            _graph_trace("before radix_attention", layer_idx)
        attn_output = self.attn(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=(not fused_q_rope and fused_kv_arg is None),
        )
        if graph_capture:
            _graph_trace("after radix_attention", layer_idx)
        output, _ = self.o_proj(attn_output)
        if graph_capture:
            _graph_trace("after o_proj", layer_idx)
        metadata.last_forward_was_decode = current_is_decode
        return output

    with profile_block(
        "mac.metadata.init_host_buffer",
        layer=layer_idx,
        forward_batch=forward_batch,
        cuda=True,
    ):
        metadata.init_host_buffer(
            hidden_states.dtype,
            hidden_states.device,
            hidden_states.shape[0],
            forward_batch.batch_size,
            getattr(server_args, "chunked_prefill_size", hidden_states.shape[0]),
            server_args,
            layer_idx,
            config,
            _active_request_capacity(
                forward_batch,
                int(getattr(server_args, "max_running_requests", forward_batch.batch_size)),
            ),
        )
    with profile_block(
        "mac.metadata.prepare_forward_batch",
        layer=layer_idx,
        forward_batch=forward_batch,
        cuda=True,
    ):
        metadata.prepare_forward_batch(forward_batch, config)

    with profile_block("llama.attention.qkv_proj", layer=layer_idx, forward_batch=forward_batch, cuda=True):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    current_is_decode = forward_batch.forward_mode.is_decode_or_idle()
    with profile_block("mac.copy_q_before_rope", layer=layer_idx, forward_batch=forward_batch, cuda=True):
        fused_q_rope = _try_mac_decode_rope_preserve(
            self, positions, q, k, v, forward_batch, metadata
        )
        if not fused_q_rope:
            _preserve_q_before_rope(metadata, q, current_is_decode, forward_batch)
    with profile_block("llama.attention.rotary_emb", layer=layer_idx, forward_batch=forward_batch, cuda=True):
        fused_kv_arg = None
        if not fused_q_rope:
            fused_kv_arg = _maybe_fused_kv_rope_arg(self, v, forward_batch)
            q, k = self.rotary_emb(
                positions,
                q,
                k,
                fused_set_kv_buffer_arg=fused_kv_arg,
            )
    metadata.rotary_emb = self.rotary_emb

    if metadata.last_forward_was_decode and not current_is_decode:
        _log_profile_if_needed(metadata, config, layer_idx)
        metadata.reset()

    metadata.num_k_head = getattr(self, "num_kv_heads", 0)

    self.attn.mac_metadata = metadata
    self.attn.mac_config = config
    with profile_block("llama.attention.radix_attention", layer=layer_idx, forward_batch=forward_batch, cuda=True):
        attn_output = self.attn(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=(not fused_q_rope and fused_kv_arg is None),
        )
    with profile_block("llama.attention.o_proj", layer=layer_idx, forward_batch=forward_batch, cuda=True):
        output, _ = self.o_proj(attn_output)
    metadata.last_forward_was_decode = current_is_decode
    return output


def _llama_attention_forward_mac_direct(
    self: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: Any,
    config: Any,
) -> Any:
    try:
        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()
    except Exception:
        server_args = None

    layer_idx = int(getattr(getattr(self, "attn", None), "layer_id", 0))
    metadata: MACMetadata = getattr(self, "mac_metadata", None)
    if metadata is None:
        metadata = MACMetadata(self.num_heads, self.num_kv_heads, self.head_dim)
        self.mac_metadata = metadata

    if not profiling_enabled():
        graph_capture = _is_cuda_graph_capturing()
        if graph_capture:
            _graph_trace(
                "direct enter "
                f"mode={getattr(forward_batch, 'forward_mode', None)} "
                f"hidden={tuple(hidden_states.shape)} "
                f"batch={getattr(forward_batch, 'batch_size', None)}",
                layer_idx,
            )
        capacity = int(config.mac_lookback_tokens_left)
        max_running = int(
            getattr(server_args, "max_running_requests", forward_batch.batch_size)
        )
        request_capacity = _active_request_capacity(forward_batch, max_running)
        cache_ready = (
            metadata.unified_query_cache is not None
            and metadata.unified_query_cache.device == hidden_states.device
            and metadata.unified_query_cache.dtype == hidden_states.dtype
            and metadata.unified_query_cache.shape[0] >= request_capacity
            and metadata.unified_query_cache.shape[1] == capacity
        )
        if cache_ready:
            metadata.device = hidden_states.device
            metadata.dtype = hidden_states.dtype
            metadata.layer_idx = layer_idx
            metadata.cur_q_len = int(hidden_states.shape[0])
            metadata.batch_size = int(forward_batch.batch_size)
        else:
            metadata.init_host_buffer(
                hidden_states.dtype,
                hidden_states.device,
                hidden_states.shape[0],
                forward_batch.batch_size,
                getattr(server_args, "chunked_prefill_size", hidden_states.shape[0]),
                server_args,
                layer_idx,
                config,
                request_capacity,
            )
        if graph_capture:
            _graph_trace("direct after init_host_buffer", layer_idx)
        metadata.prepare_forward_batch(forward_batch, config)
        if graph_capture:
            _graph_trace(
                "direct after prepare_forward_batch "
                f"is_decode={forward_batch.forward_mode.is_decode_or_idle()} "
                f"cur_kv_len={getattr(metadata, 'cur_kv_len', None)}",
                layer_idx,
            )

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if graph_capture:
            _graph_trace("direct after qkv_proj", layer_idx)
        current_is_decode = forward_batch.forward_mode.is_decode_or_idle()
        fused_q_rope = _try_mac_decode_rope_preserve(
            self, positions, q, k, v, forward_batch, metadata
        )
        if not fused_q_rope:
            _preserve_q_before_rope(metadata, q, current_is_decode, forward_batch)
        if graph_capture:
            _graph_trace(f"direct after preserve_q fused_q_rope={fused_q_rope}", layer_idx)

        fused_kv_arg = None
        if not fused_q_rope:
            fused_kv_arg = _maybe_fused_kv_rope_arg(self, v, forward_batch)
            q, k = self.rotary_emb(
                positions,
                q,
                k,
                fused_set_kv_buffer_arg=fused_kv_arg,
            )
        if graph_capture:
            _graph_trace(f"direct after rotary fused_kv={fused_kv_arg is not None}", layer_idx)
        metadata.rotary_emb = self.rotary_emb

        if metadata.last_forward_was_decode and not current_is_decode:
            _log_profile_if_needed(metadata, config, layer_idx)
            metadata.reset()

        metadata.num_k_head = getattr(self, "num_kv_heads", 0)
        self.attn.mac_metadata = metadata
        self.attn.mac_config = config
        if graph_capture:
            _graph_trace("direct before radix_attention", layer_idx)
        attn_output = self.attn(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=(not fused_q_rope and fused_kv_arg is None),
        )
        if graph_capture:
            _graph_trace("direct after radix_attention", layer_idx)
        output, _ = self.o_proj(attn_output)
        if graph_capture:
            _graph_trace("direct after o_proj", layer_idx)
        metadata.last_forward_was_decode = current_is_decode
        return output

    with profile_block(
        "mac.metadata.init_host_buffer",
        layer=layer_idx,
        forward_batch=forward_batch,
        cuda=True,
    ):
        metadata.init_host_buffer(
            hidden_states.dtype,
            hidden_states.device,
            hidden_states.shape[0],
            forward_batch.batch_size,
            getattr(server_args, "chunked_prefill_size", hidden_states.shape[0]),
            server_args,
            layer_idx,
            config,
            _active_request_capacity(
                forward_batch,
                int(getattr(server_args, "max_running_requests", forward_batch.batch_size)),
            ),
        )
    with profile_block(
        "mac.metadata.prepare_forward_batch",
        layer=layer_idx,
        forward_batch=forward_batch,
        cuda=True,
    ):
        metadata.prepare_forward_batch(forward_batch, config)

    with profile_block("llama.attention.qkv_proj", layer=layer_idx, forward_batch=forward_batch, cuda=True):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    current_is_decode = forward_batch.forward_mode.is_decode_or_idle()
    with profile_block("mac.copy_q_before_rope", layer=layer_idx, forward_batch=forward_batch, cuda=True):
        fused_q_rope = _try_mac_decode_rope_preserve(
            self, positions, q, k, v, forward_batch, metadata
        )
        if not fused_q_rope:
            _preserve_q_before_rope(metadata, q, current_is_decode, forward_batch)
    with profile_block("llama.attention.rotary_emb", layer=layer_idx, forward_batch=forward_batch, cuda=True):
        fused_kv_arg = None
        if not fused_q_rope:
            fused_kv_arg = _maybe_fused_kv_rope_arg(self, v, forward_batch)
            q, k = self.rotary_emb(
                positions,
                q,
                k,
                fused_set_kv_buffer_arg=fused_kv_arg,
            )
    metadata.rotary_emb = self.rotary_emb

    if metadata.last_forward_was_decode and not current_is_decode:
        _log_profile_if_needed(metadata, config, layer_idx)
        metadata.reset()

    metadata.num_k_head = getattr(self, "num_kv_heads", 0)
    self.attn.mac_metadata = metadata
    self.attn.mac_config = config
    with profile_block("llama.attention.radix_attention", layer=layer_idx, forward_batch=forward_batch, cuda=True):
        attn_output = self.attn(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=(not fused_q_rope and fused_kv_arg is None),
        )
    with profile_block("llama.attention.o_proj", layer=layer_idx, forward_batch=forward_batch, cuda=True):
        output, _ = self.o_proj(attn_output)
    metadata.last_forward_was_decode = current_is_decode
    return output


def _make_llama_attention_forward_direct(original_fn: Any) -> Any:
    def forward(
        self: Any,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: Any,
    ) -> Any:
        config = getattr(forward_batch, "_mac_config", None)
        if config is None:
            config = getattr(getattr(self, "attn", None), "mac_config", None)
        if config is None:
            config = get_mac_config()
        if config.enable_mac and hidden_states.device.type == "cuda":
            return _llama_attention_forward_mac_direct(
                self, positions, hidden_states, forward_batch, config
            )
        return original_fn(self, positions, hidden_states, forward_batch)

    return forward


llama_attention_forward_around.__mac_attention_direct_factory__ = (
    _make_llama_attention_forward_direct
)


def _run_decoder_layer_post_attention_cache_update(
    result: Any, self: Any, *args: Any, **kwargs: Any
) -> Any:
    metadata = getattr(getattr(self, "self_attn", None), "mac_metadata", None)
    if metadata is None:
        return result
    if not get_mac_config().enable_mac:
        return result
    forward_batch = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
    attn_backend = getattr(forward_batch, "attn_backend", None) if forward_batch is not None else None
    if hasattr(attn_backend, "launch_pending_decode_cache_update"):
        attn_backend.launch_pending_decode_cache_update(metadata)
    if metadata.wait_cache_update_after_ffn:
        metadata.wait_for_cache_update()
    return result


def llama_decoder_layer_forward_around(
    original_fn: Any, self: Any, *args: Any, **kwargs: Any
) -> Any:
    forward_batch = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
    layer_idx = int(getattr(getattr(getattr(self, "self_attn", None), "attn", None), "layer_id", 0))
    trace_enabled = os.environ.get("MAC_DEBUG_GRAPH_TRACE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    patched_modules: list[tuple[Any, Any]] = []

    def patch_module(module: Any, label: str) -> None:
        if module is None:
            return
        original_forward = getattr(module, "forward", None)
        if not callable(original_forward):
            return

        def traced_forward(*module_args: Any, **module_kwargs: Any) -> Any:
            _graph_trace(f"{label} enter", layer_idx)
            result = original_forward(*module_args, **module_kwargs)
            _graph_trace(f"{label} exit", layer_idx)
            return result

        try:
            setattr(module, "forward", traced_forward)
            patched_modules.append((module, original_forward))
        except Exception:
            return

    if trace_enabled:
        patch_module(getattr(self, "input_layernorm", None), "input_layernorm")
        patch_module(getattr(self, "self_attn", None), "self_attn")
        patch_module(getattr(self, "post_attention_layernorm", None), "post_attention_layernorm")
        patch_module(getattr(self, "mlp", None), "mlp")
    if not profiling_enabled():
        try:
            result = original_fn(self, *args, **kwargs)
            return _run_decoder_layer_post_attention_cache_update(result, self, *args, **kwargs)
        finally:
            for module, original_forward in reversed(patched_modules):
                setattr(module, "forward", original_forward)
    try:
        with profile_block(
            "llama.decoder_layer.forward",
            layer=layer_idx,
            forward_batch=forward_batch,
            cuda=True,
        ):
            result = original_fn(self, *args, **kwargs)
        with profile_block(
            "mac.decoder_layer.post_attention_cache_update",
            layer=layer_idx,
            forward_batch=forward_batch,
            cuda=True,
        ):
            return _run_decoder_layer_post_attention_cache_update(result, self, *args, **kwargs)
    finally:
        for module, original_forward in reversed(patched_modules):
            setattr(module, "forward", original_forward)


def llama_mlp_forward_around(original_fn: Any, self: Any, *args: Any, **kwargs: Any) -> Any:
    forward_batch = kwargs.get("forward_batch", args[1] if len(args) > 1 else None)
    if not profiling_enabled():
        return original_fn(self, *args, **kwargs)
    with profile_block("llama.mlp.forward", forward_batch=forward_batch, cuda=True):
        return original_fn(self, *args, **kwargs)


def llama_model_forward_around(original_fn: Any, self: Any, *args: Any, **kwargs: Any) -> Any:
    input_ids = kwargs.get("input_ids", args[0] if len(args) > 0 else None)
    positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
    forward_batch = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
    config = get_mac_config()
    if config.enable_mac and forward_batch is not None:
        if not (_is_torch_compiling() or _is_piecewise_cuda_graph_active()):
            setattr(forward_batch, "_mac_config", config)
        device = None
        if torch.is_tensor(positions):
            device = positions.device
        elif torch.is_tensor(input_ids):
            device = input_ids.device
        if device is not None:
            try:
                from sglang.srt.server_args import get_global_server_args

                server_args = get_global_server_args()
            except Exception:
                server_args = None
            _prepare_mac_forward_batch_state(forward_batch, device, config, server_args)
    if not profiling_enabled():
        return original_fn(self, *args, **kwargs)
    with profile_block("llama.model.forward", forward_batch=forward_batch, cuda=True):
        return original_fn(self, *args, **kwargs)


def _make_llama_model_forward_direct(original_fn: Any) -> Any:
    def forward(
        self: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[Any] = None,
    ) -> Any:
        return llama_model_forward_around(
            original_fn,
            self,
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    return forward


llama_model_forward_around.__mac_attention_direct_factory__ = (
    _make_llama_model_forward_direct
)


def llama_for_causal_lm_forward_around(
    original_fn: Any, self: Any, *args: Any, **kwargs: Any
) -> Any:
    forward_batch = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
    if not profiling_enabled():
        return original_fn(self, *args, **kwargs)
    with profile_block("llama.for_causal_lm.forward", forward_batch=forward_batch, cuda=True):
        return original_fn(self, *args, **kwargs)
