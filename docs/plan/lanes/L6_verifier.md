# L6_verifier


Continuous. Owns `GATES.md`, `stats.py`, `make tables`, STATE.md signatures. For every "done": re-run the lane's tests from a clean worktree; re-run `make tables` and diff against claimed numbers; `/ponytail-review` on the merged diff; forbidden-word grep; prereg SHA precedes launch SHA; manifest error count equals `error` rows in `trials.jsonl`; no arm reported as `--` where trials errored. Sign STATE.md with commands + output paths. Failure → back to the lane with `systematic-debugging` required.
