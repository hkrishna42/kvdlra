"""Streaming subspace trackers: the incremental-SVD step the decode cache runs."""

from collections.abc import Callable

from torch import Tensor

from kvdlra.tracker.isvd import fd_step, isvd_step, oja_step

__all__ = ["TRACKERS"]

# The gist trackers a cache arm can name, by the string its config carries. The three
# share the ``(u, b_core, block, rank_cap) -> (u_new, b_new, rot)`` contract and differ
# in their keyword-only knobs, so the values are typed by their return only; the caller
# knows which keywords the branch it took needs. ``Callable`` and not a Protocol because
# the whole point is that the signatures differ.
TRACKERS: dict[str, Callable[..., tuple[Tensor, Tensor, Tensor]]] = {
    "isvd": isvd_step,
    "fd": fd_step,
    "oja": oja_step,
}
