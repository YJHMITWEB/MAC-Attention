from __future__ import annotations

import os
import sys
from typing import Any, Callable


def _graph_trace(message: str) -> None:
    if os.environ.get("MAC_DEBUG_GRAPH_TRACE", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    print(f"[mac-graph-trace][cuda-graph-runner] {message}", file=sys.stderr, flush=True)


def cuda_graph_capture_one_batch_size_around(
    original_fn: Callable[..., Any], runner: Any, *args: Any, **kwargs: Any
) -> Any:
    """Trace SGLang CUDA graph capture without requiring edits to SGLang."""
    if os.environ.get("MAC_DEBUG_GRAPH_TRACE", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return original_fn(runner, *args, **kwargs)

    bs = args[0] if args else kwargs.get("bs", "?")
    _graph_trace(f"capture_one_batch_size enter bs={bs}")

    device_module = getattr(runner, "device_module", None)
    original_sync = getattr(device_module, "synchronize", None)
    patched_sync = False

    if original_sync is not None:

        def traced_sync(*sync_args: Any, **sync_kwargs: Any) -> Any:
            _graph_trace("device_module.synchronize enter")
            result = original_sync(*sync_args, **sync_kwargs)
            _graph_trace("device_module.synchronize exit")
            return result

        try:
            setattr(device_module, "synchronize", traced_sync)
            patched_sync = True
        except Exception as exc:  # pragma: no cover - best-effort debug hook
            _graph_trace(f"device_module.synchronize patch failed: {exc}")

    tp_group = getattr(getattr(runner, "model_runner", None), "tp_group", None)
    original_barrier = getattr(tp_group, "barrier", None)
    patched_barrier = False

    if original_barrier is not None:

        def traced_barrier(*barrier_args: Any, **barrier_kwargs: Any) -> Any:
            _graph_trace("tp_group.barrier enter")
            result = original_barrier(*barrier_args, **barrier_kwargs)
            _graph_trace("tp_group.barrier exit")
            return result

        try:
            setattr(tp_group, "barrier", traced_barrier)
            patched_barrier = True
        except Exception as exc:  # pragma: no cover - best-effort debug hook
            _graph_trace(f"tp_group.barrier patch failed: {exc}")

    patched_args = args
    patched_kwargs = kwargs

    forward_fn = args[1] if len(args) >= 2 else kwargs.get("forward")
    if callable(forward_fn):

        def traced_forward(*forward_args: Any, **forward_kwargs: Any) -> Any:
            _graph_trace("forward enter")
            result = forward_fn(*forward_args, **forward_kwargs)
            _graph_trace("forward exit")
            return result

        if len(args) >= 2:
            patched_args = (args[0], traced_forward, *args[2:])
        else:
            patched_kwargs = dict(kwargs)
            patched_kwargs["forward"] = traced_forward

    try:
        result = original_fn(runner, *patched_args, **patched_kwargs)
        _graph_trace(f"capture_one_batch_size exit bs={bs}")
        return result
    finally:
        if patched_sync:
            setattr(device_module, "synchronize", original_sync)
        if patched_barrier:
            setattr(tp_group, "barrier", original_barrier)
