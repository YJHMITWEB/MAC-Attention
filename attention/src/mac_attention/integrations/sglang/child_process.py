from __future__ import annotations

import os
from typing import Any


def run_scheduler_process_with_mac_hooks(*args: Any, **kwargs: Any) -> Any:
    """Spawn-safe SGLang scheduler entry point that installs MAC hooks in child."""
    from .hook_installer import install_hooks

    installed = install_hooks()
    if os.environ.get("MAC_ATTENTION_SGLANG_DEBUG") == "1":
        print(
            "[MAC-Attention] scheduler child hooks installed: "
            f"{len(installed)} new hook(s)",
            flush=True,
        )

    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)
