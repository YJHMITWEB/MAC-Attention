import argparse
import json
import os

from mac_attention.integrations.sglang.config import (
    MAC_CONFIG_ENV,
    MacSGLangConfig,
    get_mac_config,
    install_env_config,
)
from mac_attention.integrations.sglang.plugin import server_args_add_cli_args_around


def _noop_add_cli_args(parser):
    parser.add_argument("--official-arg", default="ok")
    return parser


def test_env_only_config_uses_best_so_far_defaults(monkeypatch):
    monkeypatch.delenv(MAC_CONFIG_ENV, raising=False)
    monkeypatch.setenv("MAC_ATTENTION_ENABLE", "1")

    cfg = MacSGLangConfig.from_env()

    assert cfg is not None
    assert cfg.enable_mac
    assert cfg.mac_threshold == 0.45
    assert cfg.mac_lookback_tokens_left == 512
    assert cfg.mac_lookback_tokens_right == 0
    assert cfg.mac_semantic_pos_ahead == 256
    assert cfg.mac_persistent_tile_tokens == 32
    assert cfg.mac_persistent_partial_fp32 is True


def test_env_aliases_override_json_transport(monkeypatch):
    monkeypatch.setenv(
        MAC_CONFIG_ENV,
        json.dumps(
            {
                "enable_mac": False,
                "mac_threshold": 0.7,
                "mac_persistent_tile_tokens": 64,
                "mac_persistent_partial_fp32": False,
            }
        ),
    )
    monkeypatch.setenv("MAC_ENABLE", "1")
    monkeypatch.setenv("MAC_THRESHOLD", "0.45")
    monkeypatch.setenv("MAC_PERSISTENT_TILE_TOKENS", "32")
    monkeypatch.setenv("MAC_PERSISTENT_PARTIAL_FP32", "1")

    cfg = MacSGLangConfig.from_env()

    assert cfg is not None
    assert cfg.enable_mac
    assert cfg.mac_threshold == 0.45
    assert cfg.mac_persistent_tile_tokens == 32
    assert cfg.mac_persistent_partial_fp32 is True


def test_install_env_config_materializes_child_process_json(monkeypatch):
    monkeypatch.delenv(MAC_CONFIG_ENV, raising=False)
    monkeypatch.setenv("MAC_ATTENTION_ENABLE", "1")
    monkeypatch.setenv("MAC_PERSISTENT_TILE_TOKENS", "32")
    monkeypatch.setenv("MAC_PERSISTENT_PARTIAL_FP32", "1")

    cfg = install_env_config()
    child_cfg = json.loads(os.environ[MAC_CONFIG_ENV])

    assert cfg.enable_mac
    assert child_cfg["enable_mac"] is True
    assert child_cfg["mac_persistent_tile_tokens"] == 32
    assert child_cfg["mac_persistent_partial_fp32"] is True


def test_get_mac_config_cache_tracks_env_changes(monkeypatch):
    monkeypatch.delenv(MAC_CONFIG_ENV, raising=False)
    monkeypatch.setenv("MAC_ATTENTION_ENABLE", "1")
    server_args = object()
    assert get_mac_config(server_args).enable_mac is True

    monkeypatch.setenv("MAC_ATTENTION_ENABLE", "0")
    assert get_mac_config(server_args).enable_mac is False


def test_plugin_does_not_add_mac_cli_args_by_default(monkeypatch):
    monkeypatch.delenv("MAC_ATTENTION_EXPOSE_CLI_ARGS", raising=False)
    parser = argparse.ArgumentParser()

    server_args_add_cli_args_around(_noop_add_cli_args, parser)

    option_strings = set(parser._option_string_actions)
    assert "--official-arg" in option_strings
    assert "--enable-mac" not in option_strings
    assert "--mac-threshold" not in option_strings


def test_plugin_can_expose_legacy_mac_cli_args_for_debug(monkeypatch):
    monkeypatch.setenv("MAC_ATTENTION_EXPOSE_CLI_ARGS", "1")
    parser = argparse.ArgumentParser()

    server_args_add_cli_args_around(_noop_add_cli_args, parser)

    option_strings = set(parser._option_string_actions)
    assert "--official-arg" in option_strings
    assert "--enable-mac" in option_strings
    assert "--mac-threshold" in option_strings
