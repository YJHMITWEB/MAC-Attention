#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_mac_portable.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MAC_HOOK_CHECK_STRICT="${MAC_HOOK_CHECK_STRICT:-1}"
export MAC_HOOK_CHECK_STRICT

"$PYTHON_BIN" - <<'PY'
import os

from mac_attention.integrations.sglang.config import install_env_config
from mac_attention.integrations.sglang.hook_installer import install_hooks, is_installed

strict = os.environ.get("MAC_HOOK_CHECK_STRICT", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
cfg = install_env_config()
installed = install_hooks(strict=strict)
print("enable_mac", cfg.enable_mac)
print("tile_tokens", cfg.mac_persistent_tile_tokens)
print("partial_fp32", cfg.mac_persistent_partial_fp32)
print("installed_hooks", len(installed))
print("server_args_hooked", is_installed())
PY
