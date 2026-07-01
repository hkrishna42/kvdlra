"""Test the kvpress transformers>=5.8 compatibility shim (hermetic; no model)."""

from __future__ import annotations

from kvdlra.press.compat import install_kvpress_prefill_compat


def test_install_patches_base_press_and_is_idempotent() -> None:
    from kvpress.presses.base_press import BasePress

    original = BasePress.forward_hook
    install_kvpress_prefill_compat()
    patched = BasePress.forward_hook
    assert patched is not original  # forward_hook was replaced

    install_kvpress_prefill_compat()  # idempotent: no further change
    assert BasePress.forward_hook is patched
