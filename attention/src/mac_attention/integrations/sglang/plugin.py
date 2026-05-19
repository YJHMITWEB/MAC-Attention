from __future__ import annotations

import os
from typing import Any

from .config import MacSGLangConfig, add_cli_args, get_mac_config, install_on_server_args


def _bool_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def server_args_add_cli_args_around(original_fn: Any, parser: Any) -> Any:
    result = original_fn(parser)
    if _bool_env("MAC_ATTENTION_EXPOSE_CLI_ARGS"):
        add_cli_args(parser)
    return result


def server_args_from_cli_args_after(result: Any, cls: Any, args: Any) -> Any:
    config = MacSGLangConfig.from_namespace(args)
    return install_on_server_args(result, config)


def server_args_post_init_after(result: Any, self: Any) -> Any:
    config = get_mac_config(self)
    if config.enable_mac:
        install_on_server_args(self, config)
    return result


def register_plugin() -> None:
    strict = _bool_env("MAC_ATTENTION_SGLANG_STRICT", "1")
    try:
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("sglang.srt.plugins"):
            from .hook_installer import install_hooks

            install_hooks(strict=strict)
            return
        raise

    from .hook_installer import _resolve_target, is_installed, iter_hook_specs

    # The portable launcher installs hooks directly before SGLang loads its
    # plugin entry points. In that mode, registering the same hooks with
    # HookRegistry would wrap hot-path methods twice.
    if is_installed():
        return

    hook_types = {
        "around": HookType.AROUND,
        "after": HookType.AFTER,
    }
    for spec in iter_hook_specs():
        if spec.optional:
            try:
                parent, attr_name = _resolve_target(spec.target)
            except Exception:
                continue
            if not hasattr(parent, attr_name):
                continue
        HookRegistry.register(spec.target, spec.hook, hook_types[spec.kind])
