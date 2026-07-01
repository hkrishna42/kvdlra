"""Compatibility shim: make stock kvpress presses work on transformers >= 5.8.

kvpress ``BasePress.forward_hook`` detects the pre-fill phase via
``kwargs["cache_position"]``, but transformers >= 5.8 does **not** thread
``cache_position`` into the attention module's kwargs (verified in Week 3: the
module receives only ``hidden_states``, ``attention_mask``, ``past_key_values``,
``position_embeddings``, ``position_ids``, ``use_cache``). So any stock press
(SnapKV, ExpectedAttention, ...) ``KeyError``s on a plain ``model(...)`` forward.

:func:`install_kvpress_prefill_compat` replaces ``BasePress.forward_hook`` with a
byte-for-byte copy of the upstream logic (including the ``QuantizedCache`` branch)
except the pre-fill check falls back to ``q_len > 1`` when ``cache_position`` is
absent. It changes **no compression behaviour** — only how pre-fill is detected —
so head-to-head baselines (SnapKV/ExpectedAttention/...) are unaffected in
substance. :class:`kvdlra.press.BUGPress` overrides ``forward_hook`` itself and is
unaffected by this patch.

Call once before running any stock press under our harness. Idempotent.
"""

from __future__ import annotations

from typing import Any

from torch import nn

_PATCHED = False


def install_kvpress_prefill_compat() -> None:
    """Patch ``kvpress.BasePress.forward_hook`` for transformers >= 5.8 (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return

    from kvpress.presses.base_press import BasePress
    from kvpress.utils import extract_keys_and_values
    from transformers import QuantizedCache

    def forward_hook(
        self: BasePress,
        module: nn.Module,
        input: list[Any],
        kwargs: dict[str, Any],
        output: list[Any],
    ) -> list[Any]:
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        cache_layer = cache.layers[module.layer_idx]
        q_len = hidden_states.shape[1]

        cache_position = kwargs.get("cache_position")
        if cache_position is not None:
            if cache_position[-1] > q_len:  # past pre-fill (upstream semantics)
                return output
        elif q_len <= 1:  # transformers>=5.8: no cache_position -> detect by q_len
            return output

        keys, values = extract_keys_and_values(cache, module.layer_idx)
        keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)

        if isinstance(cache, QuantizedCache):
            cl = cache_layer
            cl._quantized_keys = cl._quantize(keys, axis=cl.axis_key)
            cl._quantized_values = cl._quantize(values, axis=cl.axis_value)
            cl.keys = keys.new_zeros(0)
            cl.values = keys.new_zeros(0)
            cl.cumulative_length = keys.shape[2]
        else:
            cache_layer.keys = keys
            cache_layer.values = values
        return output

    BasePress.forward_hook = forward_hook
    _PATCHED = True
