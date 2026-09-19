"""The kvpress presses an arm YAML's ``press:`` block names (L2.4, ruling PR-L2-12).

Five families: three token-eviction scorers at a ``keep`` fraction (``snapkv``,
``pyramidkv``, ``expected_attention``; kvpress's ``compression_ratio`` is ``1 - keep``),
ThinK's key-channel pruning at a ``ratio`` (``think``), and the two composed as ThinK's
paper evaluates it -- evict first, then prune the survivors' channels (``think_snapkv``,
``keep`` + ``ratio``). PyramidKV runs at kvpress 0.5.1's defaults (window 64, kernel 5,
beta 20).

``family:`` is explicit on every NEW arm. The archived arms carry none -- ``doc:`` and
every key are part of the pod config hash their archive pods pin -- so for them the family
is inferred from the config name prefix exactly as frontier's former ``_evict_factory`` did
(``snapkv*`` -> snapkv, ``think*`` -> think, else expected_attention), and a ``family:``
that contradicts that inference is refused rather than trusted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kvdlra.eval.config import ArmCfg

if TYPE_CHECKING:
    from kvpress import BasePress

FAMILIES = ("snapkv", "pyramidkv", "expected_attention", "think", "think_snapkv")


def press_family(cfg: ArmCfg) -> str | None:
    """The family of ``cfg``'s press, or None when the block names no kvpress press (the
    ``full`` arm, or the SVD oracle's ``rank`` block, which frontier's ``_press`` owns)."""
    p = cfg.press
    if not ("keep" in p or "ratio" in p):
        return None
    inferred = _inferred(cfg.name, p)
    fam = str(p.get("family", inferred))
    if fam not in FAMILIES:
        raise ValueError(f"configs/arms/{cfg.name}.yaml: press.family {fam!r} not in {FAMILIES}")
    if fam != inferred:
        raise ValueError(
            f"configs/arms/{cfg.name}.yaml: press.family {fam!r} contradicts the name prefix"
            f" ({inferred!r})"
        )
    return fam


def _inferred(name: str, p: dict[str, Any]) -> str:
    if "ratio" in p:
        return "think_snapkv" if "keep" in p else "think"
    return next((f for f in ("snapkv", "pyramidkv") if name.startswith(f)), "expected_attention")


def make_press(cfg: ArmCfg) -> BasePress | None:
    """A fresh press for ``cfg`` (presses are stateful: one per sample), or None."""
    from kvpress import (
        ComposedPress,
        ExpectedAttentionPress,
        PyramidKVPress,
        SnapKVPress,
        ThinKPress,
    )

    fam = press_family(cfg)
    if fam is None:
        return None
    p = cfg.press
    if fam == "think":
        return ThinKPress(key_channel_compression_ratio=float(p["ratio"]))
    if fam == "think_snapkv":
        return ComposedPress(
            [
                SnapKVPress(compression_ratio=1.0 - float(p["keep"])),
                ThinKPress(key_channel_compression_ratio=float(p["ratio"])),
            ]
        )
    scorer = {
        "snapkv": SnapKVPress,
        "pyramidkv": PyramidKVPress,
        "expected_attention": ExpectedAttentionPress,
    }[fam]
    return scorer(compression_ratio=1.0 - float(p["keep"]))
