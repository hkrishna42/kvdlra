# Week-19 kickoff prompt: close the gap — fair-quant fork, official anchor, arXiv v1 → ICML 2027

*Paste the block below into a fresh Claude Code session in this repo. It is self-contained. It
encodes the state at the end of the Week-18 GPU program (2026-09-05), the exit-gate panel verdict
(5/6 dimensions ≥7, zero fatal, significance 6), the expert readiness verdict, and the ranked,
costed program to close the remaining gap. Full facts/gotchas appendix below the paste block.*

**Honesty bar (carried):** no process guarantees acceptance. Exit bar = *every* panel dimension
≥7 with zero fatal; arXiv v1 live; ICML-2027-ready draft. We are one decisive experiment short.

---

> **Week-19: resume the kvdlra paper program. Everything through the Week-18 GPU matrix is DONE;
> your job is to close the last gap, finalize the paper, and ship arXiv v1 → ICML 2027.
> Orchestrate all GPU pods yourself (never sub-agents); use parallel agents only for $0 work
> (review, citations, figures).**
>
> **Start by reading** (in order): the `kvdlra-week18-g1` memory (full state + gotchas),
> `docs/week19-kickoff.md` §Facts (below), `results/w18-g1-report.md`, `paper/main.tex`
> (the arXiv-v1 draft, GPU-confirmed, 780 lines), `results/w18_harvest/quant-findings.md`
> (the quant rabbit hole), and `docs/reviews/2026-09-01/review-meta-verdict.md` (the original
> panel). Then `git log --oneline -25` on `week7`.
>
> ## Where we are (one paragraph)
> Phases 0–2 of the Week-18 program are complete and pushed: the wording/billing/evidence pass,
> the harness engineering (quant arm + dual billing + per-trial emission + realistic filler +
> eviction/marquee/persistence MODEs), and the **full G1–G5 GPU matrix** (3 families,
> SHA-pinned pods, unattended watchdog). The manuscript is GPU-confirmed across 8 sections and
> citation-clean (34 refs verified, 5 fixed). The **exit-gate panel** (6 adversarial dimension
> reviewers + AC) scored: claims 7, prior-work 7, rigor 7, systems 7 (was 3), repro 7,
> **significance 6** — zero fatal, up from 5/10 borderline-reject. The $0 fix pass from that
> panel is applied (paired McNemar marquee, xKV novelty sharpening, 1/T framing, env/SHA
> provenance, peak_ratio footnote). Credit ≈ **$94**; no pods running.
>
> ## The verdict you are executing against
> **arXiv v1: ready today. NeurIPS/ICML main track: strong borderline (~6), not yet.** The
> *idea* and *rigor* are top-venue; the *empirical case* is mixed — BUG matches retrieval,
> loses fluency, loses resident memory (1.06× full), wins stored/persisted bytes and one hard
> cell (marquee vt, McNemar-separated from ThinK p=0.002 and Palu p=0.031), and its exclusive
> sub-0.05× band is **conditional** on a fair 2-bit baseline never run.
>
> **The crux — one experiment decides the paper's shape.** The killer reviewer question is not
> "does 2-bit quant retrieve?" but *"at matched stored bytes, why not just quantize?"* KIVI-2bit
> sits at ~0.125× stored, next to our 0.085–0.149×, and ALSO wins resident memory. So the fair
> KIVI baseline determines whether the contribution is **broad** (BUG retrieves where 2-bit
> quant fails → exclusive band real → clear accept) or **narrow** (2-bit also retrieves → BUG is
> a sub-0.05× storage-tier method → honest poster on mechanism+rigor). **Run it first; write the
> ICML version around its outcome. Do not write around it.**
>
> ## Phase A — the four gap-closing experiments (ranked by leverage; ≈$105 total)
>
> **A1 (DECISIVE, ~$30): a fair KIVI/KVQuant 2-bit baseline.** What exists: `quant-{2,4}bit`
> arms via transformers 5.8 `QuantizedCache(backend="quanto", nbits, axis_key, axis_value,
> q_group_size=64, residual_length=128)`, accounting verified on GPU (0.194×/0.318×), flags
> `--quant-nbits/--quant-group/--quant-residual/--quant-axis-key/--quant-axis-value`, MODEs
> `g1quant` + `g1diag` in `scripts/pod/w18.sh`. What's broken (all measured on Qwen-7B @16K):
> the **default per-channel** config gives **0.00 retrieval** on 2 AND 4-bit while full=1.00
> and BUG=1.00 (correct accounting, so not a setup artifact); the KIVI-faithful per-token axis
> `--quant-axis-value=-1` SKIPs with `Group size (64) must be a divisor of (65588)`; quanto
> supports **only 2/4-bit** (no 8-bit control); quant **ppl OOMs at 32K even on 80GB**.
> Fix plan, validate on ONE `-devel` diag pod before fanning out (the decisive signal is a
> `quant-*bit acc=` ROW, not `QUANTO_OK`): (i) per-token values — find a `q_group_size` that
> divides the per-token axis (try head_dim-aligned: 128 for Llama/Mistral, 128 for Qwen; or
> group = head_dim) OR switch backend to **HQQ** (`backend="HQQ"`, supports 1/2/3/4/8-bit and
> per-axis quant natively — also gives the **8-bit control**: if 8-bit scores 0 the decode path
> is buggy, if 8-bit retrieves and 2-bit doesn't the loss is real); (ii) memory-safe quant ppl —
> smaller `--window`, fewer `--n-samples`, or score in chunks; (iii) then run the grid: 2/4-bit
> (+8-bit control) × 3 families × 16K/32K × RULER{single,mk,mv,vt} + ppl, n=12, plus the
> **BUG×quant compose** (`bugS-r64-h256-q4` is the existing coord-quant path; note
> `bug_cache.py:477-481` forbids `seed_hh_warmup ⊕ quant` — the seeded compose needs that guard
> relaxed + tested). **Fund bar:** report WHATEVER it shows (compose-not-compete framing). If
> BUG holds single+mv ≥0.7 Wilson-lo where 2-bit ≤0.3 → the exclusive band is claimed; else the
> band claim is retired and §subcliff/§limits reworded to the narrow story.
>
> **A2 (~$50): an official-benchmark anchor.** Official RULER (NVIDIA repo) niah subset on the
> flagship, one model (Llama), + the `scripts/w10_longbench.py` flagship thread-through (its
> argparse is missing `--min-sv-frac`/the 4 flagship args; it imports `build_arms` so add them
> like `w10_ruler.py:496-502`). Publish the generator-vs-official template diff. Closes the
> "self-authored benchmark" reject-reason that WikiText filler (g2) only half-closed.
>
> **A3 (~$10): a realized systems win.** Extend `scripts/w16_storage.py` (CUDA, Llama-8B,
> 16K/32K) with an end-to-end **serialize → reload → H2D → reconstruct-to-attend-ready
> wall-clock** for full-KV vs `bugSseed-r64-h256` vs the 2-bit baseline. The byte ratio
> (cold-load 0.150×/0.139×) is measured; the wall-clock is what turns it into a deployment
> existence proof and stops "compression paper with no realized win."
>
> **A4 (~$15): push the 1/T asymptotic with data.** The gist is O(rn+hn), constant in T, so the
> ratio falls as 1/T (0.149→0.139, 0.085→0.075 from 16K→32K) — the ONE claim no fixed-bit
> quantizer can ever match. Add a **64K** (ideally 128K; needs 80GB) point: flagship (use
> r128+`--min-sv-frac 1e-2` for Qwen at ≥32K — its r64 ppl blows up 8→35 there) vs `ea-k0.1`
> (its multikey decays with length: 92→67→50) vs 2-bit. Cheap, uniquely ours, and it broadens
> the n=1 marquee. Escalate this to a headline figure.
>
> Order: **A1 → A4 → A2 → A3.** Pre-register each block's MODE in `w18.sh` (successor `w19.sh`),
> `bash -n` + CPU-smoke every new flag through the real harness, push before launching.
>
> ## Phase B — finish the paper ($0, parallelizable)
> 1. **Figures**: `paper/main.tex` still points at the stale `figures/week4/hero.pdf`.
>    Regenerate the frontier/retrieval/1-over-T figures from the committed W18 line-files
>    (dataviz/graphing skill); add the eviction grid + storage table as figures.
> 2. **LaTeX build**: no TeX toolchain on this Mac (`pdflatex`/`latexmk` absent). Add a CI
>    latex job or build on a machine with TeX; fix any overfull/undefined on first build. All
>    `\ref/\label/\cite` currently resolve by grep.
> 3. **Update §subcliff/§quantbaseline/§limits/abstract** with A1's outcome (broad vs narrow),
>    §memory with A3's wall-clock, and add the A4 1/T figure. Low-priority cites the panel
>    flagged: Frequent Directions (Liberty), Oja, incremental SVD, ShadowKV (named but uncited),
>    the merging family (CaM/KVMerger), and note the MomentKV surprise-signal overlap + priority.
> 4. **Dashboard** artifact `757d6777` — update with the W18 grids (pass `url:`).
> 5. **Polish**: prose through elements-of-style; `/code-review` on all harness diffs.
> 6. **Re-run the exit-gate panel** (same 6-dimension + AC protocol; reviewer prompts are in
>    this session's transcript pattern) after A1 lands. Bar: ≥7 every dimension, zero fatal.
>
> ## Phase C — ship
> **arXiv v1 now** (the honest current story is submittable today; it timestamps vs MomentKV
> 2606.01563 / ResKV 2607.29591) — user approves the submission. Then the **ICML 2027** version
> with A1–A4 folded in, framed by A1's outcome.
>
> ## Infra playbook (proven this session — reuse verbatim)
> - **Launch**: `vastai create instance <offer> --image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-{runtime|devel}
>   --disk 80 --onstart <scratch copy of scripts/pod/w18_boot.sh with MODE/MODEL/TAG/SHA sed-baked> --label kvdlra-w19-<tag>`.
>   **Quant arms need the `-devel` image** (quanto JIT-builds `quanto_cuda.so`; `-runtime` has
>   no nvcc → `CUDA_HOME not set` SKIP). Everything else runs on cheaper `-runtime`. Bake the
>   SHA into the onstart (`sed 's|${SHA:-week7}|${SHA:-<sha>|'`) — pods clone by SHA, print
>   `===RUN_SHA_<sha>===` + nvidia-smi + torch/CUDA into the log.
> - **Offers**: `vastai search offers 'gpu_name=A100_PCIE num_gpus=1 disk_space>120 inet_down>300 reliability>0.98' -o dph --raw`;
>   PCIe 40GB ≈$0.60–1.1/hr (80GB for 8B@32K+). Only ~4–8 offers at a time — launch one at a
>   time, check each contract. Credit from `vastai show user --raw` (`credit` field; the CLI
>   Balance column lies). Destroy needs `echo y | vastai destroy instance <id>`.
> - **Validate one pod first, then fan out** — and the validation signal is a **result row**
>   (`[niah_single ctx16384] <arm> acc=… n=…`), not a boot marker. We lost two iterations
>   fanning out on `QUANTO_OK`.
> - **Harvest incrementally**: the vastai log buffer scrolls (~400 lines) under the per-`[trial]`
>   emission, and the base64 JSON folds arrive TRUNCATED — so harvest the short aggregate rows
>   (`acc=`, `ppl=`, `[pplw`, `stored_ratio=`, `[trial]`) every ~2–3 min into
>   `results/w19_harvest/<label>.raw`, dedup with `sort -u` at extraction. Do NOT rely on an
>   end-of-run `vastai logs`. Do not commit the multi-MB raws (gitignored); commit the deduped
>   line-files + `results/w19_pertrial/` + an env-provenance extract (`RUN_SHA|A100|cuda_build`).
> - **Unattended completion**: run the watchdog **detached** via a python double-fork
>   (`os.fork(); os.setsid(); os.fork(); execvp bash`) so PPID=1 and it survives Claude
>   teardown/terminal close (plain `run_in_background` tasks die on teardown; macOS has no
>   `setsid`, no `mapfile`, no `declare -A` — bash 3.2). Per pod: harvest → destroy on its own
>   `===ALL_DONE_<mode>_<tag>` → extract → commit → push; credit-floor (~$6) destroys all.
>   **Give the long-pole pod its own fallback with a bigger loop budget** — watchdog2's 16.7h
>   budget expired before g4 (~22h) finished; the g4 fallback (26h) caught it. Do NOT restart a
>   watchdog mid-run (its in-memory done-list of destroyed pods is unrecoverable → it hangs
>   waiting for pods whose logs are gone). Tie `caffeinate -is -w <pid>` to it.
> - **Timing (measured)**: 16K streaming RULER cell ≈ 20–30 min; **32K r64 cell ≈ 1.4h;
>   32K r256 cell ≈ 2h**; presses (think/palu/ea/snapkv) are fast; g4 (marquee n=16 + r256 +
>   same-pod ppl) took ~22h. Size cells before launching; `PPL4` (n-samples 4) not `PPL`.
> - **Harness rules**: never drop `--chunk` (the RULER()/PPL() wrappers pass it); every new flag
>   default-off + bit-identical-off test + fail-loud tripwire + CPU probe on 1B; `_plot` now
>   survives all-SKIP runs; `w16_storage._smoke_args` drift fixed via `build_parser().parse_args([])`.
>
> ## Gotchas (carried, hard-won)
> - The eviction baseline is **family-dependent**: `ea-k0.1` @0.1× is strong on Llama
>   single/mk/mv (loses vt) but collapses on Qwen/Mistral — and collapses to 0.00 on ALL tasks
>   under WikiText filler (g2). Never quote one family as "eviction."
> - The `q4` compose arm shows an odd 16K→32K inversion (mv/vt 0.00→1.00 on Qwen/Llama) —
>   understand it before citing q4.
> - Qwen r64 @32K: retrieval 1.00 but ppl 8→35 — use r128+floor at ≥32K for Qwen.
> - Marquee stat is the **paired McNemar** (shared `seed*131+trial` needles), not Fisher; the
>   multiple-comparisons defense is *pre-registration (family=1)*, not Bonferroni.
> - Two program SHAs: g1 = `15678a7`, g2–g5 = `b157acd` (harness identical; quant-axis + plot
>   diffs only). Keys unrotated (flag only).
>
> ## Process rules
> Green-gate every increment (`uv run pytest -q && ruff check . && mypy --strict src tests scripts`),
> commit per increment, push before pods, pre-register every matrix + fund bar before spending,
> harvest-before-destroy, escalate below $3 credit, report results WHATEVER they show.

---

## §Facts (end of Week-18, 2026-09-05)

- **Branch** `week7`, latest `fd93586` (exit-gate fix pass). Key commits: `e882405` Phase-0/1;
  `15678a7` quant install fix (g1 SHA); `b157acd` g1 results (g2–g5 SHA); `05038e3` quant-axis
  flags + g1diag; `62aa785` paper GPU-results; `904ca59` refs fixed; `9b46ada` marquee same-pod;
  `296d3ed` per-trial records; `fd93586` McNemar/xKV/1-over-T/provenance.
- **Results** (committed): `results/w18-g1-report.md`; line-files `results/w18-{qwen,mistral,llama}-lines.txt`
  (g1), `w18-g2-qwen`, `w18-g3-{qwen,mistral,llama}`, `w18-g4-llama`, `w18-g5-llama-lines.txt`;
  Wilson `results/w18-g1-*-ruler-intervals.{json,md}`; McNemar `results/w18-g4-marquee-contrasts.json`;
  per-trial `results/w18_pertrial/*-trials.txt` (1842 lines); env `results/w18-env-provenance.txt`;
  quant `results/w18_harvest/quant-findings.md`. Panel reviews `docs/reviews/2026-09-01/`.
- **Headline numbers** (n=12 unless noted): flagship `bugSseed-r64-h256` 16K single/mk/mv 1.00
  all families, vt Qwen 1.00 / Llama 0.58 / Mistral 0.50 @0.149×/0.085×; 32K single/mk 1.00,
  mv 1.00 (Mistral 0.83), vt Qwen 1.00 / Llama 0.92 / Mistral 0.42 @0.139×/0.075×. Marquee
  `bugSseed-r128-h1024-s32` @32K n=16: vt 0.94 vs think 0.31 (McNemar p=0.0020) / palu 0.56
  (p=0.031); same-pod ppl full 6.975 / think 7.196 / palu 7.232 / flagship 7.353. r256 wall
  control 0.00 on all four tasks. g5: stored 0.084×, cold-load 0.150×, workspace 0.982× @16K.
- **Scripts**: `scripts/pod/w18_boot.sh` (SHA-pinned onstart, env header, optimum-quanto,
  CUDA_HOME), `scripts/pod/w18.sh` (MODEs g1/g1quant/g1diag/g2/g3/g4/g5), `w18_watchdog{,2}.sh`,
  `w18_g4fallback.sh`, `w18_finalizer.sh`, `scripts/w18_intervals.py` (Wilson + McNemar,
  reads `n=`), `scripts/w10_ruler.py` (per-trial `[trial]` lines, `--filler`, `--depths`,
  quant flags), `scripts/w10_frontier.py` (`build_arms`, `build_parser`, quant arm + accounting).
- **Dashboard** artifact `757d6777`. Manuscript `paper/main.tex` + `paper/refs.bib` + `Makefile`.
- **Budget**: credit ≈$94.36 at end of session; the whole W18 GPU program cost ≈$76.
