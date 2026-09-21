# Gate-1 panel prompt — five simulated reviewers read the Stage-1 table before the team does

Pre-registration: `prereg/gate1_tracker_swap_v2.md` §10. Table: `docs/paper/tables/gate1.md`,
rendered by `make gate1` from `results/gate1_v2_stage1_{llama,qwen}/`. This file is the prompt the
orchestrator pastes; it is **not** an input to the panel and no reader is given it.

## 1. Why this exists

`prereg/gate1_tracker_swap_v2.md` §10, lines 1195–1200, verbatim:

> - **The five-reviewer simulation runs on the table before anyone reads it.**
>   `docs/plan/KICKOFF_WEEKS0-3.md` Part E: "When Gate 1 v2 harvests, run the five-reviewer
>   simulation on the table alone before reading it yourself." The simulation sees
>   `docs/paper/tables/gate1.md` and this file, and nothing else — no lane report, no summary, no
>   branch recommendation — and its output is committed beside the table. Only then is the verdict
>   read.

`docs/plan/KICKOFF_WEEKS0-3.md` Part E (heading at line 318), line 327, last sentence, verbatim:

> When Gate 1 v2 harvests, run the five-reviewer simulation on the table alone before reading it
> yourself.

The rule this document enforces, in the order it binds:

1. `make gate1` renders the table with its stdout discarded, and the table is committed **unread**.
2. The orchestrator dispatches five readers **without having opened** `docs/paper/tables/gate1.md`.
   Nothing about what the table shows can therefore enter a dispatch prompt: the orchestrator does
   not know it. The prompts below are written from the renderer's code, never from its output.
3. The five readings, and `meta.md`, are committed **beside the table**, under
   `docs/plan/reviews/<YYYY-MM-DD>-gate1/`.
4. **Only then** is the `VERDICT:` line read, and the Gate-1 outcome appended to
   `docs/plan/DECISIONS.md`.

A reading produced after someone on the team has said what the table shows is not the reading §10
asks for, and committing it later does not repair it.

## 2. What every reader gets, and nothing else

Exactly two files:

- `docs/paper/tables/gate1.md` — the table `make gate1` rendered from
  `results/gate1_v2_stage1_{llama,qwen}/`.
- `prereg/gate1_tracker_swap_v2.md` — the pre-registration, with every dated amendment appended
  below §11.

Forbidden inputs, named so a reader cannot drift into them: `results/` in any form, `docs/plan/STATE.md`,
`docs/plan/DECISIONS.md`, `docs/plan/cleanup/`, `docs/plan/reports/`, `docs/plan/SESSION_PROMPT_*.md`,
`docs/plan/lanes/`, `docs/plan/plans/`, this file, `git log` and any other git history, running
`make gate1` or any other code, any other `prereg/*.md`, any other table under `docs/paper/`, and
the paper. A reader who needs a number the two files do not carry writes **"not in the table"** and
moves on — it is a finding, not an errand.

One consequence to state now, because the table invites it: ahead of the verdict the table carries a
**bf16 block** and a `BF16:` line whose rule lives in `prereg/bf16_gist.md`, which is **not** an
input (§10's letter).
A reader reports the bf16 rows as printed and writes "rule not in the two files" rather than judging
them; that reading is taken from its own file afterwards, by the orchestrator.

**Quoting a cell.** In a retrieval block (`## <family> — ctx <ctx>`) the row is a **tracker**
(`full`, `isvd`, `nogist`, `frozen`, `fd`, `bf16`, `oja`, `random` — the `<!-- arms: … -->` comment
above the block maps each to its arm file name) and the column is a **task** (`niah_single`,
`niah_multikey`, `niah_multivalue`, `vt`); a cell reads `acc [Wilson 95% lo,hi] (hits/n)`, or
`FAILED (k errors)`, or `not run (…)` with the reason — never `--`. A `*` or `‡` **sits on the
control's row**, because the member is `isvd` against that control: `*` is a Holm-significant
primary contrast favouring `isvd`, `‡` one favouring the control. Quote row, column and the cell
text.

## 3. The five lenses

Each reader takes ONE lens. All five answer the same five questions (§4) through their lens.

**R1 — Pre-registration compliance and statistics.** Does every printed statistic correspond to one
§4/§6 names — the exact paired McNemar over the shared `(seed, trial)` keys, Holm at α = 0.05 over
the families §6 fixes **as amended**, the paired *t*-test, the paired bootstrap CI and the ±0.02
TOST over the 32 windows? Are the **realised** family sizes the pre-registered ones? They are printed
in the header comment as `primary retrieval m=…, secondary retrieval m=…, primary perplexity m=…`;
check them against §6 (16 / 24 / 4 with no task excluded, 12 / 18 / 4 under an amendment that
excludes one task, and §6's arm-drop and task-exclusion composition rule if both fired). Are dropped
keys, error rows and refusals reported as §4 and Amendment 1a require — dropped keys as the
`- pairing: k key(s) dropped (…) -- a vs b, n_paired N` bullets, error rows as `FAILED (k errors)`
cells plus their `- <arm> / <task>: k error records, first …` bullets, refusals inside the `VERDICT:`
reason and repeated in `members:`, with `REFUSED` distinct from `UNDECIDED` (A1a.2) and a refused
family out of **every** Holm family before the correction (A1a.3 — its primary perplexity rows then
print `refused (excluded from the Holm family)` where an adjusted p would be)? **Recompute one Holm
adjustment and one TOST by hand.** The raw McNemar p-values are not printed, so take the adjustment
from a `members:` line that carries its pair counts (`(a-b pairs, Holm p=…)`): recompute the exact
two-sided p as §6 writes it, `2·P(Bin(a+b, ½) ≤ min(a, b))`, place it in the Holm slot for the
printed realised m, and say whether the printed adjusted p agrees. Take the TOST from a perplexity
row's printed `95% CI` using §6's own read-back, `s ≈ (hi − lo)·√n / 3.92` at the pre-registered
n = 32, check it against §6's decidability bound (`s < 0.0667` at ±0.02) and say whether the printed
`passes` / `fails` / `not decidable` agrees. Does the `VERDICT:` line follow from its `members:` by
the rule exactly as §4 writes it — A/B only if **both** families are separated from **both** `frozen`
and `nogist`; C only if no Holm-significant `fd`/`frozen` separation exists on any task in any family
**and** every `fd`/`frozen` TOST passes; otherwise `REFUSED` or `UNDECIDED`?

**R2 — Mechanism.** Reading `isvd` against `frozen`, `random`, `nogist`, `fd` and `oja`: where do
retrieval and perplexity come from? Does the byte-matched no-gist arm carry retrieval? Is the frozen
basis as good as the tracked one? Does the `fd` arm separate anywhere — it is one of C's two
controls, so a separation prints as a `members:` line `<family>/<task>: isvd vs fd separates (Holm
p=…)`, while `oja` and `random` carry **no** adjusted retrieval p anywhere in the table and are read
from their accuracy cells and their perplexity rows alone (say so where it limits you). Compare the
pattern with the predictions §5 wrote before the run, arm by arm, and say where the table agrees and
where it does not. What would you conclude about whether the online-tracked gist does work a frozen
basis or no gist does not?

**R3 — Experimental validity.** n = 24 per cell (the `(hits/n)` in each cell; a member's `n_paired`
can be smaller, and the pairing bullets say by how much and on which keys): what can and cannot be
separated at that n, against §6's 24-key resolution note? Floor-against-floor cells — two cells at
the same low accuracy with no `*` and no `‡`, which §4 fixes as McNemar p = 1.0 at zero discordance,
an ordinary p-value and never evidence of equivalence. The `vt` rows: a primary member where no
amendment excludes the task, **descriptive only** under the amendment that excludes it — the printed
realised m says which (12 / 18 means a task left the families), and the §6 exclusion rule says what
follows. The real-document haystacks. The pairing report. The no-gist arms' byte match — §4 reads it
from `sbits` and prints a ratio only when it **refuses** a family, so say whether the table lets you
check it at all. Any error row. The `full` row as the ceiling, and §6 (ii)'s flag where a `full` cell
sags below 0.9 on a task the pre-flight passed. The two-family structure: one separated family reads
`UNDECIDED` with `one family separated (<family>)` as its reason, never rounded up (§4 rule 2). What would a sceptical reviewer demand before
accepting the branch?

**R4 — Claims and writing.** List the sentences a paper could write from this table alone, and the
sentences it could not. Say whether "the tracker is load-bearing" is writable. Name every overclaim a
reviewer would attack, and the neutral wording that survives. Note any number the table prints that a
reader could mistake for a memory, throughput or baseline claim: the table makes none, and the two
places that look like one are the bf16 block's `sbits bf16/isvd` column and a byte-match refusal
line's stored-bits ratio — both pins on whether an arm ran as specified, neither a result. Say
whether a reader would take them that way.

**R5 — Area chair.** Read the whole table. Which branch does a fair reader infer, independent of the
`VERDICT:` line? What is the single strongest objection? What must Stage 2 (Mistral 16K; the 32K
pods) show to change or confirm it — noting §4's scope, that the verdict reads the 16K contrasts only
and any 32K block in the table is descriptive? Give a 1–10 confidence that the online-tracked gist
does work a frozen basis or no gist does not, and a 1–10 confidence that the table follows its
pre-registration.

Two reading rules every lens shares, because the table's own structure sets them. The perplexity
blocks (`### <family> — perplexity, ctx <ctx>[, <corpus>]`) print, per tracker row:
`bits/token`, `delta (isvd - tracker)` — negative is the r64 arm ahead — `95% CI`, `TOST +/-0.02`
(three states: `passes`, `fails`, `not decidable`), `Holm p` (`n/a (secondary)` outside the 4-member
primary family, `refused (excluded from the Holm family)` for a primary member whose family was
refused) and `paired t p`. And the `VERDICT:` line's `members:` list is where the per-member pair
counts, adjusted p-values, refusal texts and C-blockers are printed in full — the retrieval blocks
carry accuracies and marks, not member statistics.

## 4. What every reader returns — one page, ≤ 60 lines, this shape

1. **Numbers I read** — every cell you rely on, quoted with its row (tracker) and column (task, or
   the perplexity column name), or the `members:` line you took it from.
2. **Rule compliance as I read it** — the rule clause, the members that decide it, agree/disagree
   with the `VERDICT:` line.
3. **The three strongest objections.**
4. **What I would need to see before believing the branch.**
5. **One-line verdict in my own words.**

File: `docs/plan/reviews/<YYYY-MM-DD>-gate1/R<k>-<slug>.md`, where `<slug>` is
`compliance` | `mechanism` | `validity` | `claims` | `chair`.

## 5. How the orchestrator dispatches it

Five Agent calls in ONE message, `subagent_type` general-purpose, model opus, `run_in_background`
true, each prompt = the preamble below + one lens from §3 + the return format of §4 + its output
path. `<YYYY-MM-DD>` is the dispatch date, the same for all five. The orchestrator writes nothing
about the table into any prompt, because it has not read the table. When all five have returned, the
orchestrator writes `docs/plan/reviews/<YYYY-MM-DD>-gate1/meta.md` (≤ 15 lines: objections raised by
≥ 2 readers; disagreements between readers; whether R1's compliance check failed and on what),
commits the directory together with the table, and **only then** reads the `VERDICT:` line and
appends the Gate-1 outcome to `docs/plan/DECISIONS.md`.

### Preamble (verbatim in every dispatch; fill only the three bracketed paths)

```text
You are one of five independent reviewers reading a pre-registered experiment's result table blind.
You may open exactly two files: <path to gate1.md> and <path to prereg/gate1_tracker_swap_v2.md>.
Open nothing else — no results directory, no plan or state or decisions files, no git history, no
other prereg, and run no code. Treat the table as the only evidence and the prereg as the only rule.
Do not summarise the prereg back; read it to check the table against it. Write your reading to
<output path> in the five-part shape below, ≤ 60 lines, numbers quoted with their cell. Return only
the output path.
```

## 6. After the panel

The readings are evidence for the DECISIONS entry, not a vote: the branch is `gate1_verdict()`'s. The
entry cites the readings' path beside `results/gate1_v2_stage1_{llama,qwen}/` and
`docs/paper/tables/gate1.md`. If R1 reports a compliance failure — a printed statistic that is not
the pre-registered one — that is a harness defect to fix in a later commit and a re-render, never a
reason to read the verdict differently.
