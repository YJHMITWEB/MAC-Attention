from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import inspect
from importlib import import_module
from importlib import metadata
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Iterable, Literal


HookKind = Literal["around", "after"]


@dataclass(frozen=True)
class HookSpec:
    target: str
    hook: Callable[..., Any]
    kind: HookKind
    optional: bool = False


def _profiling_enabled() -> bool:
    return os.environ.get("MAC_ATTENTION_SGLANG_PROFILE", "0").strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    }


def _graph_trace_enabled() -> bool:
    return os.environ.get("MAC_DEBUG_GRAPH_TRACE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def iter_hook_specs() -> Iterable[HookSpec]:
    from .flashinfer_hooks import (
        flashinfer_backend_init_after,
        flashinfer_forward_decode_around,
        flashinfer_forward_extend_around,
        flashinfer_init_forward_metadata_capture_cuda_graph_around,
        flashinfer_init_forward_metadata_around,
        flashinfer_init_forward_metadata_replay_cuda_graph_around,
    )
    from .cuda_graph_hooks import cuda_graph_capture_one_batch_size_around
    from .llama_hooks import (
        llama_attention_forward_around,
        llama_attention_init_after,
        llama_decoder_layer_forward_around,
        llama_for_causal_lm_forward_around,
        llama_mlp_forward_around,
        llama_model_forward_around,
    )
    from .plugin import (
        server_args_add_cli_args_around,
        server_args_from_cli_args_after,
        server_args_post_init_after,
    )
    from .schedule_hooks import (
        forward_batch_init_new_after,
        schedule_get_model_worker_batch_after,
        schedule_mix_with_running_after,
        schedule_prepare_for_extend_after,
    )

    yield HookSpec(
        "sglang.srt.server_args:ServerArgs.add_cli_args",
        server_args_add_cli_args_around,
        "around",
    )
    yield HookSpec(
        "sglang.srt.server_args:ServerArgs.from_cli_args",
        server_args_from_cli_args_after,
        "after",
    )
    yield HookSpec(
        "sglang.srt.server_args:ServerArgs.__post_init__",
        server_args_post_init_after,
        "after",
    )

    yield HookSpec(
        "sglang.srt.managers.schedule_batch:ScheduleBatch.prepare_for_extend",
        schedule_prepare_for_extend_after,
        "after",
    )
    yield HookSpec(
        "sglang.srt.managers.schedule_batch:ScheduleBatch.mix_with_running",
        schedule_mix_with_running_after,
        "after",
    )
    yield HookSpec(
        "sglang.srt.managers.schedule_batch:ScheduleBatch.get_model_worker_batch",
        schedule_get_model_worker_batch_after,
        "after",
        optional=True,
    )
    yield HookSpec(
        "sglang.srt.model_executor.forward_batch_info:ForwardBatch.init_new",
        forward_batch_init_new_after,
        "after",
    )

    yield HookSpec(
        "sglang.srt.models.llama:LlamaAttention.__init__",
        llama_attention_init_after,
        "after",
    )
    yield HookSpec(
        "sglang.srt.models.llama:LlamaAttention.forward",
        llama_attention_forward_around,
        "around",
    )
    if _profiling_enabled() or _graph_trace_enabled():
        yield HookSpec(
            "sglang.srt.models.llama:LlamaDecoderLayer.forward",
            llama_decoder_layer_forward_around,
            "around",
        )
    if _profiling_enabled():
        yield HookSpec(
            "sglang.srt.models.llama:LlamaMLP.forward",
            llama_mlp_forward_around,
            "around",
        )
    yield HookSpec(
        "sglang.srt.models.llama:LlamaModel.forward",
        llama_model_forward_around,
        "around",
    )
    if _profiling_enabled():
        yield HookSpec(
            "sglang.srt.models.llama:LlamaForCausalLM.forward",
            llama_for_causal_lm_forward_around,
            "around",
        )

    yield HookSpec(
        "sglang.srt.layers.attention.flashinfer_backend:FlashInferAttnBackend.__init__",
        flashinfer_backend_init_after,
        "after",
    )
    yield HookSpec(
        "sglang.srt.layers.attention.flashinfer_backend:FlashInferAttnBackend.init_forward_metadata",
        flashinfer_init_forward_metadata_around,
        "around",
    )
    yield HookSpec(
        "sglang.srt.layers.attention.flashinfer_backend:FlashInferAttnBackend.init_forward_metadata_capture_cuda_graph",
        flashinfer_init_forward_metadata_capture_cuda_graph_around,
        "around",
        optional=True,
    )
    yield HookSpec(
        "sglang.srt.layers.attention.flashinfer_backend:FlashInferAttnBackend.init_forward_metadata_replay_cuda_graph",
        flashinfer_init_forward_metadata_replay_cuda_graph_around,
        "around",
        optional=True,
    )
    yield HookSpec(
        "sglang.srt.layers.attention.flashinfer_backend:FlashInferAttnBackend.forward_decode",
        flashinfer_forward_decode_around,
        "around",
    )
    yield HookSpec(
        "sglang.srt.layers.attention.flashinfer_backend:FlashInferAttnBackend.forward_extend",
        flashinfer_forward_extend_around,
        "around",
    )
    if _graph_trace_enabled():
        yield HookSpec(
            "sglang.srt.model_executor.cuda_graph_runner:CudaGraphRunner.capture_one_batch_size",
            cuda_graph_capture_one_batch_size_around,
            "around",
            optional=True,
        )


def _resolve_target(target: str) -> tuple[Any, str]:
    module_name, qualname = target.split(":", 1)
    obj = import_module(module_name)
    parts = qualname.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    return obj, parts[-1]


def _descriptor_function(raw_descriptor: Any) -> Any:
    if isinstance(raw_descriptor, (classmethod, staticmethod)):
        return raw_descriptor.__func__
    return raw_descriptor


def _call_hook(
    kind: HookKind,
    hook: Callable[..., Any],
    original_fn: Callable[..., Any],
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
) -> Any:
    if kind == "around":
        return hook(original_fn, *call_args, **call_kwargs)

    result = original_fn(*call_args, **call_kwargs)
    replacement = hook(result, *call_args, **call_kwargs)
    return result if replacement is None else replacement


def _make_wrapper(
    target: str,
    kind: HookKind,
    hook: Callable[..., Any],
    original_fn: Callable[..., Any],
) -> Callable[..., Any]:
    direct_factory = getattr(hook, "__mac_attention_direct_factory__", None)
    if direct_factory is not None:
        wrapped = direct_factory(original_fn)
        wrapped.__mac_attention_hooked__ = True
        wrapped.__mac_attention_hook_target__ = target
        wrapped.__mac_attention_hook_kind__ = kind
        wrapped.__mac_attention_original__ = original_fn
        return wrapped

    @wraps(original_fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if kind == "around":
            return hook(original_fn, *args, **kwargs)
        result = original_fn(*args, **kwargs)
        replacement = hook(result, *args, **kwargs)
        return result if replacement is None else replacement

    wrapped.__mac_attention_hooked__ = True
    wrapped.__mac_attention_hook_target__ = target
    wrapped.__mac_attention_hook_kind__ = kind
    wrapped.__mac_attention_original__ = original_fn
    return wrapped


def _install_hook(spec: HookSpec) -> bool:
    parent, attr_name = _resolve_target(spec.target)
    raw_descriptor = parent.__dict__.get(attr_name)
    if raw_descriptor is None:
        raw_descriptor = getattr(parent, attr_name, None)
    if raw_descriptor is None:
        if spec.optional:
            return False
        raise AttributeError(f"{parent!r} has no attribute {attr_name!r}")

    current_fn = _descriptor_function(raw_descriptor)
    if getattr(current_fn, "__mac_attention_hooked__", False):
        return False

    wrapped = _make_wrapper(spec.target, spec.kind, spec.hook, current_fn)
    if isinstance(raw_descriptor, classmethod):
        replacement = classmethod(wrapped)
    elif isinstance(raw_descriptor, staticmethod):
        replacement = staticmethod(wrapped)
    else:
        replacement = wrapped
    setattr(parent, attr_name, replacement)
    return True


def _install_child_process_wrappers() -> tuple[str, ...]:
    """Make spawned SGLang scheduler processes install MAC hooks before model load."""
    from .child_process import run_scheduler_process_with_mac_hooks

    installed: list[str] = []
    engine = import_module("sglang.srt.entrypoints.engine")
    if (
        getattr(engine, "run_scheduler_process", None)
        is not run_scheduler_process_with_mac_hooks
    ):
        engine.run_scheduler_process = run_scheduler_process_with_mac_hooks
        installed.append("sglang.srt.entrypoints.engine:run_scheduler_process")
    engine_cls = getattr(engine, "Engine", None)
    if engine_cls is not None and (
        getattr(engine_cls, "run_scheduler_process_func", None)
        is not run_scheduler_process_with_mac_hooks
    ):
        engine_cls.run_scheduler_process_func = staticmethod(
            run_scheduler_process_with_mac_hooks
        )
        installed.append("sglang.srt.entrypoints.engine:Engine.run_scheduler_process_func")

    http_server = import_module("sglang.srt.entrypoints.http_server")
    if (
        getattr(http_server, "run_scheduler_process", None)
        is not run_scheduler_process_with_mac_hooks
    ):
        http_server.run_scheduler_process = run_scheduler_process_with_mac_hooks
        installed.append("sglang.srt.entrypoints.http_server:run_scheduler_process")
    launch_server = getattr(http_server, "launch_server", None)
    if launch_server is not None and _replace_default_argument(
        launch_server,
        "run_scheduler_process_func",
        run_scheduler_process_with_mac_hooks,
    ):
        installed.append(
            "sglang.srt.entrypoints.http_server:launch_server.run_scheduler_process_func"
        )

    data_parallel = import_module("sglang.srt.managers.data_parallel_controller")
    if (
        getattr(data_parallel, "run_scheduler_process", None)
        is not run_scheduler_process_with_mac_hooks
    ):
        data_parallel.run_scheduler_process = run_scheduler_process_with_mac_hooks
        installed.append(
            "sglang.srt.managers.data_parallel_controller:run_scheduler_process"
        )
    run_data_parallel = getattr(data_parallel, "run_data_parallel_controller_process", None)
    if run_data_parallel is not None and _replace_default_argument(
        run_data_parallel,
        "run_scheduler_process_func",
        run_scheduler_process_with_mac_hooks,
    ):
        installed.append(
            "sglang.srt.managers.data_parallel_controller:"
            "run_data_parallel_controller_process.run_scheduler_process_func"
        )

    return tuple(installed)


def _replace_default_argument(
    fn: Callable[..., Any], parameter_name: str, replacement: Callable[..., Any]
) -> bool:
    defaults = fn.__defaults__
    if not defaults:
        return False

    params = [
        param
        for param in inspect.signature(fn).parameters.values()
        if param.default is not inspect.Parameter.empty
        and param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
    ]
    names = [param.name for param in params]
    try:
        default_idx = names.index(parameter_name)
    except ValueError:
        return False
    if defaults[default_idx] is replacement:
        return False
    new_defaults = list(defaults)
    new_defaults[default_idx] = replacement
    fn.__defaults__ = tuple(new_defaults)
    return True


def _debug_enabled() -> bool:
    return os.environ.get("MAC_ATTENTION_SGLANG_DEBUG", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _git_commit_for_path(path: Path) -> str:
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            try:
                return subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "--short", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                return "unknown"
    return "not-a-git-checkout"


def _diagnostic_lines() -> tuple[str, ...]:
    lines: list[str] = []
    try:
        import sglang

        sglang_path = Path(getattr(sglang, "__file__", "")).resolve()
        try:
            sglang_version = metadata.version("sglang")
        except metadata.PackageNotFoundError:
            sglang_version = "editable-or-uninstalled"
        lines.append(
            "[MAC-Attention] SGLang: "
            f"version={sglang_version}, path={sglang_path}, "
            f"git={_git_commit_for_path(sglang_path)}"
        )
    except Exception as exc:
        lines.append(f"[MAC-Attention] SGLang: unavailable ({exc})")

    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(device)
            device_name = torch.cuda.get_device_name(device)
            cuda_info = (
                f"available=True, device={device}, name={device_name}, "
                f"capability=sm_{capability[0]}{capability[1]}"
            )
        else:
            cuda_info = "available=False"
        lines.append(f"[MAC-Attention] CUDA: {cuda_info}")
    except Exception as exc:
        lines.append(f"[MAC-Attention] CUDA: unavailable ({exc})")

    try:
        from .bridge import _torch_ext_build_dir

        lines.append(f"[MAC-Attention] extension build dir: {_torch_ext_build_dir()}")
    except Exception as exc:
        lines.append(f"[MAC-Attention] extension build dir: unavailable ({exc})")

    return tuple(lines)


def install_hooks(*, strict: bool = True) -> tuple[str, ...]:
    """Install MAC-Attention hooks into SGLang without editing SGLang files."""
    installed: list[str] = []
    errors: list[str] = []

    if _debug_enabled():
        for line in _diagnostic_lines():
            print(line, flush=True)

    for spec in iter_hook_specs():
        try:
            if _install_hook(spec):
                installed.append(spec.target)
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    "MAC-Attention SGLang plugin incompatible with this SGLang "
                    f"checkout: failed hook {spec.target}"
                ) from exc
            errors.append(f"{spec.target}: {exc}")

    try:
        installed.extend(_install_child_process_wrappers())
    except Exception as exc:
        if strict:
            raise RuntimeError(
                "MAC-Attention SGLang plugin incompatible with this SGLang "
                "checkout: failed child-process hook installation"
            ) from exc
        errors.append(f"child process wrapper: {exc}")

    if errors and strict:
        raise RuntimeError("; ".join(errors))
    if _debug_enabled():
        print(
            "[MAC-Attention] hook installer: "
            f"{len(installed)} installed, {len(errors)} error(s)",
            flush=True,
        )
        for target in installed:
            print(f"[MAC-Attention]   installed {target}", flush=True)
        for error in errors:
            print(f"[MAC-Attention]   skipped {error}", flush=True)
    return tuple(installed)


def is_installed() -> bool:
    parent, attr_name = _resolve_target(
        "sglang.srt.server_args:ServerArgs.add_cli_args"
    )
    raw_descriptor = parent.__dict__.get(attr_name)
    if raw_descriptor is None:
        raw_descriptor = getattr(parent, attr_name)
    return bool(
        getattr(_descriptor_function(raw_descriptor), "__mac_attention_hooked__", False)
    )
