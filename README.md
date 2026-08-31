# OncoHarness

A harness for putting a language model to work on cancer detection **without
trusting it**. The LLM plans, weighs evidence, and decides — but everything it
sees is verified, everything it does is recorded, and everything it becomes is
gated. Research software; not a medical device.

---

## The concepts

**The model is not the system.** A raw model asked "is there cancer here?" is a
single unaccountable guess. The harness turns that guess into a *procedure*:
fixed steps, verified evidence, an audit trail, and a measured decision. The
intelligence is rented; the discipline is structural.

**Rails, not freedom.** A deterministic state machine drives every case through
the same stations: ingest → quality check → detect → verify → aggregate →
adjudicate → report. The LLM acts at exactly one station — adjudication. It
cannot reorder the pipeline, skip verification, or wander. Same case in, same
route through.

**Detectors propose, the LLM adjudicates.** Specialist vision models run first,
tuned for high sensitivity: they over-propose candidate findings. The LLM's job
is judgment — weigh the verified evidence for each candidate and decide recall,
no-recall, or defer. Detection recall is the hard ceiling of the whole system;
the LLM can only filter, never conjure.

**The LLM never sees pixels and never authors a number.** Images live in an
artifact store and move by *handle* — opaque references. Every measurement,
coordinate, and score comes from a deterministic tool. If a number appears in a
report, a tool computed it; the LLM only chose to include it. This kills
hallucinated sizes, invented coordinates, and confident descriptions of images
never actually seen.

**Verification is symmetric and independent.** A false-positive hunter tries to
kill each candidate (and must name a specific benign alternative); a
false-negative hunter re-searches the regions nobody proposed, with a different
detector family. One direction alone drifts: an FP-hunter-only system slowly
learns to miss cancer. Generator and checker are never the same process —
self-verification is a mirage.

**Abstention is an answer.** Confidence comes from structure — conformal
prediction sets, ensemble disagreement, out-of-distribution distance — never
from the model saying "I'm confident." When evidence is thin, the case defers
to a human *with its evidence attached*. A system that cannot say "I don't
know" is not measuring itself.

**Every claim lands in a ledger.** Each tool call, each piece of evidence, each
decision appends to a hash-chained log. Change one entry and the chain breaks.
An auditor can replay any case end-to-end: what was seen, what was computed,
what was decided, what it cost.

**Cost is a first-class measurement.** Every case is metered — tool calls,
pixels touched, LLM tokens, dollars, wall time. The metric that matters is not
accuracy but accuracy *per unit of compute*: sensitivity per dollar, per
thousand tool calls, per minute. What isn't measured silently bloats.

**The LLM node is budgeted.** Prompts are cache-aligned (stable prefix, compact
evidence payload); each adjudication has a token and dollar ceiling; on breach,
error, or malformed output, a deterministic rule-based adjudicator answers
instead — degrading toward deferral, never toward a guess. Every adjudication
writes a typed trace: inputs, decision, cost, and whether the fallback fired.

**Self-improving, never self-certifying.** The harness may propose changes to
itself — new thresholds, new prompts, new policies. It may never approve them.
Promotion requires passing a conjunctive eval gate whose rules live in a path
the harness cannot write. The optimizer can trigger an evaluation; it cannot
author the exam.

**Improvement is a ratchet, not a drift.** The machinery that makes the loop
safe:

- **Policy registry** — every configuration is content-hashed (its identity
  *is* its hash), recorded in a lineage chain with its parent. One pointer
  marks the champion. Rollback is one command to any ancestor.
- **Failure bank** — every confirmed failure becomes a permanent regression
  case. A candidate that re-breaks anything the champion passed is rejected,
  whatever its average score. The bank only grows; the ratchet has teeth.
- **Sequential acceptance** — an anytime-valid statistical test (e-process)
  that stays honest under unlimited proposals and continuous monitoring. The
  loop cannot promote noise by testing repeatedly until luck cooperates; a
  no-better-than-champion candidate is accepted at most α of the time, ever.
- **Efficiency floors** — a candidate that wins accuracy by doubling cost per
  case fails the gate. Improvement means better *and* affordable.
- **Proposal audit** — every gate evaluation is counted in an append-only log.
  How many shots the optimizer took is itself a recorded, inspectable number.

## How a case flows

```
DICOM in
  └─ canonicalize (one loader, training == serving)
  └─ preflight QC ──── junk? → reject with reason
  └─ detect (high-sensitivity proposals)
  └─ TTA self-consistency → zoom re-verify at native resolution
  └─ FP hunt (named alternative) + FN hunt (second family, blind spots)
  └─ aggregate verified evidence
  └─ ADJUDICATE ← the one LLM station (text + handles only, budgeted, traced)
  └─ conformal deferral policy
  └─ report {decision | defer + evidence}, cost stamped, ledger sealed
```

## How an improvement cycle flows

```
propose      candidate policy (thresholds / prompts / routing), content-hashed,
             registered with parent lineage
evaluate     replay failure bank → run eval suite → stream paired outcomes
             into the sequential test        (all triggered, none authored,
                                              by the optimizer)
gate         conjunctive: non-inferiority + subgroup floors + calibration +
             determinism + zero bank regressions + efficiency floors +
             sequential acceptance           (rules in a path the harness
                                              cannot write)
promote      champion pointer moves; report stamped with policy id
  │
  └─ regress in production? → rollback(one command) → failure enters the bank
                              → that mistake is now impossible to repeat
```

## How to apply it

The harness is task-agnostic scaffolding; cancer detection is its first
tenant. To put it over your own detection task:

1. **Bring a detector.** Anything satisfying the `Detector` protocol —
   `propose(pixels) -> [Candidate(box, score)]` — tuned for sensitivity.
   A deliberately boring reference implementation ships in `reference/`.
2. **Bring an encoder + head** for crop-level scoring (`reference/` shows the
   shape: embed → calibrated logistic head). Real models plug in behind the
   same two functions.
3. **Wire the toolbelt.** `Toolbelt(store, ledger)` exposes the verified
   operations; add task-specific tools there — inside the tool is the only
   place allowed to touch pixels or produce numbers.
4. **Choose the adjudicator.** `RuleBasedAdjudicator` (deterministic, free) or
   `LLMAdjudicator` (Claude Agent SDK, budgeted, traced, falls back to the
   rule-based one). Swapping them is one constructor argument — that swap *is*
   the with/without-LLM ablation.
5. **Write your gate rules** (`gates/gate_rules.yaml`): metrics, floors,
   margins, α, cost ratios. Put them somewhere the runtime cannot write.
6. **Run cases** through `HarnessPipeline.run_case`; run improvement cycles
   through the policy registry + gate. Never move the champion pointer by
   hand.

```python
from oncoharness.ledger import EvidenceLedger
from oncoharness.state_machine import HarnessPipeline
from oncoharness.store import ArtifactStore
from oncoharness.tools import Toolbelt

pipeline = HarnessPipeline(Toolbelt(ArtifactStore("runs/demo/artifacts"),
                                    EvidenceLedger("runs/demo/ledger.jsonl")))
report = pipeline.run_case(case_id, pixels)   # decision | defer, cost, ledgered
```

The design axioms behind every choice here — and the pitfall register of ways
systems like this fool themselves — live in [TASKSHEET.md](TASKSHEET.md).
