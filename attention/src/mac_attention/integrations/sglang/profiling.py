from __future__ import annotations

import atexit
import csv
import json
import os
from contextlib import contextmanager
from pathlib import Path
import threading
import time
from typing import Any, Iterator, Optional

import torch


_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_LOCK = threading.Lock()
_STATS: dict[str, dict[str, Any]] = {}
_EVENT_COUNT = 0
_REGISTERED_ATEXIT = False


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUE_VALUES


def enabled() -> bool:
    return _env_bool("MAC_ATTENTION_SGLANG_PROFILE")


def sync_cuda_enabled() -> bool:
    return _env_bool("MAC_ATTENTION_SGLANG_PROFILE_SYNC", "1")


def _profile_dir() -> Path:
    root = os.environ.get("MAC_ATTENTION_SGLANG_PROFILE_DIR")
    if root:
        return Path(root)
    workspace = Path(os.environ.get("MAC_WORKSPACE_BASE", os.getcwd()))
    return workspace / "profiles" / "sglang_components"


def _case_name() -> str:
    return os.environ.get("MAC_ATTENTION_SGLANG_PROFILE_CASE", "unknown")


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", "0"))


def _mode_name(forward_batch: Any = None) -> str:
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None:
        return "unknown"
    for name, value in (
        ("decode", "is_decode_or_idle"),
        ("extend", "is_extend"),
        ("draft_extend", "is_draft_extend"),
        ("target_verify", "is_target_verify"),
    ):
        fn = getattr(mode, value, None)
        if callable(fn):
            try:
                if bool(fn()):
                    return name
            except Exception:
                pass
    return str(mode)


def mode_name(forward_batch: Any = None) -> str:
    return _mode_name(forward_batch)


def _percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])


def _summary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, stat in sorted(_STATS.items()):
        count = int(stat["count"])
        cpu_samples = stat["cpu_samples"]
        cuda_samples = stat["cuda_samples"]
        cpu_total = float(stat["cpu_total_ms"])
        cuda_total = float(stat["cuda_total_ms"])
        rows.append(
            {
                "case": _case_name(),
                "rank": _rank(),
                "pid": os.getpid(),
                "component": key,
                "count": count,
                "cpu_total_ms": round(cpu_total, 6),
                "cpu_mean_ms": round(cpu_total / count if count else 0.0, 6),
                "cpu_p50_ms": round(_percentile(cpu_samples, 50), 6),
                "cpu_p90_ms": round(_percentile(cpu_samples, 90), 6),
                "cpu_p99_ms": round(_percentile(cpu_samples, 99), 6),
                "cuda_total_ms": round(cuda_total, 6),
                "cuda_mean_ms": round(cuda_total / count if count else 0.0, 6),
                "cuda_p50_ms": round(_percentile(cuda_samples, 50), 6),
                "cuda_p90_ms": round(_percentile(cuda_samples, 90), 6),
                "cuda_p99_ms": round(_percentile(cuda_samples, 99), 6),
            }
        )
    return rows


def flush() -> None:
    if not enabled():
        return
    with _LOCK:
        rows = _summary_rows()
    out_dir = _profile_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_case_name()}_rank{_rank()}_pid{os.getpid()}"
    csv_path = out_dir / f"{stem}_summary.csv"
    json_path = out_dir / f"{stem}_summary.json"
    fields = [
        "case",
        "rank",
        "pid",
        "component",
        "count",
        "cpu_total_ms",
        "cpu_mean_ms",
        "cpu_p50_ms",
        "cpu_p90_ms",
        "cpu_p99_ms",
        "cuda_total_ms",
        "cuda_mean_ms",
        "cuda_p50_ms",
        "cuda_p90_ms",
        "cuda_p99_ms",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)


def _ensure_atexit() -> None:
    global _REGISTERED_ATEXIT
    if _REGISTERED_ATEXIT:
        return
    atexit.register(flush)
    _REGISTERED_ATEXIT = True


def _record(component: str, cpu_ms: float, cuda_ms: Optional[float]) -> None:
    global _EVENT_COUNT
    _ensure_atexit()
    cuda_value = 0.0 if cuda_ms is None else float(cuda_ms)
    with _LOCK:
        stat = _STATS.setdefault(
            component,
            {
                "count": 0,
                "cpu_total_ms": 0.0,
                "cuda_total_ms": 0.0,
                "cpu_samples": [],
                "cuda_samples": [],
            },
        )
        stat["count"] += 1
        stat["cpu_total_ms"] += float(cpu_ms)
        stat["cuda_total_ms"] += cuda_value
        stat["cpu_samples"].append(float(cpu_ms))
        if cuda_ms is not None:
            stat["cuda_samples"].append(cuda_value)
        _EVENT_COUNT += 1
        interval = int(os.environ.get("MAC_ATTENTION_SGLANG_PROFILE_FLUSH_INTERVAL", "512"))
        should_flush = interval > 0 and (_EVENT_COUNT % interval == 0)
    if should_flush:
        flush()


def component_name(name: str, *, layer: Any = None, mode: str = "unknown") -> str:
    layer_part = "all" if layer is None else str(layer)
    return f"{mode}|layer={layer_part}|{name}"


@contextmanager
def profile_block(
    name: str,
    *,
    layer: Any = None,
    forward_batch: Any = None,
    mode: Optional[str] = None,
    cuda: bool = True,
) -> Iterator[None]:
    if not enabled():
        yield
        return

    mode_value = mode or _mode_name(forward_batch)
    component = component_name(name, layer=layer, mode=mode_value)
    use_cuda = cuda and torch.cuda.is_available()
    start_event = end_event = None
    if use_cuda:
        try:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        except Exception:
            start_event = end_event = None
            use_cuda = False

    cpu_start = time.perf_counter()
    try:
        yield
    finally:
        cpu_ms = (time.perf_counter() - cpu_start) * 1000.0
        cuda_ms: Optional[float] = None
        if use_cuda and start_event is not None and end_event is not None:
            try:
                end_event.record()
                if sync_cuda_enabled():
                    end_event.synchronize()
                cuda_ms = float(start_event.elapsed_time(end_event))
            except Exception:
                cuda_ms = None
        _record(component, cpu_ms, cuda_ms)
