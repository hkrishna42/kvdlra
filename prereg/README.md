# Pre-registration

One `prereg/<pod>.md` per pod, stating what will be run and what result would count as
which outcome, **committed before the commit that launches the pod** -- strictly before:
a pre-registration committed by the launch commit itself is refused.

`scripts/pod.py launch` refuses to create an instance unless the pod's prereg exists, is
committed, and its first commit is a strict ancestor of HEAD; `scripts/pod.py check`
re-checks that order against the manifest's `git_sha` for every harvested pod.
