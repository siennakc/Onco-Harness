# Oncoscope — Build Tasksheet (repo copy)

> Working document · task IDs are stable · source: 10-agent Opus research workflow
> (6 lanes + completeness critique + 3 gap-fills · 510 techniques surveyed) · 2026-08-29
> Canonical rendered version lives in the Claude artifact "Oncoscope Tasksheet".

An LLM harness that orchestrates vision models to detect cancer in medical images,
wrapped in a recursive self-improvement loop that gets measurably better every cycle
without fooling itself.

## Part 1 · Design axioms

Non-negotiable. When a shortcut conflicts with an axiom, the axiom wins.

- **A1 — Resolution before reasoning.** Most VLM misses are perceptual: the lesion is destroyed by encoder downsampling before attention runs. Crop-and-re-encode at native resolution is the single largest gain.
- **A2 — Detector proposes, VLM adjudicates.** Specialists run high-sensitivity/low-precision; the VLM filters FPs with tool-grounded checks. Detector recall is the hard ceiling (MedRAX 63.1% vs GPT-4o 56.4%).
- **A3 — The LLM never authors pixels, coordinates, or numbers.** Every measurement, mask, and score comes from a deterministic tool into an append-only evidence ledger. The LLM plans, routes, contextualizes, synthesizes — and never sees raw pixels (handle-passing store).
- **A4 — Generator and checker must be independent.** Self-verification is a mirage (~38% false-verification documented). Verifiers: different model family, new tool calls, perceptual questions.
- **A5 — Verification is symmetric.** An FP-hunter alone drifts toward missing cancer; pair it with an FN-hunter fed blind-spot regions.
- **A6 — The evaluation harness is the product.** Locked, hash-sealed, query-budgeted test sets, conjunctive gates, and regression suites come before any model work.
- **A7 — Only three channels inject new information.** Human adjudication, clinician reports, clinical outcomes. Pseudo-labels and RLAIF only sharpen existing beliefs.
- **A8 — Improvement is a ratchet.** Auto-growing regression suite, per-subgroup non-inferiority gates, lineage with rehearsed one-command rollback.
- **A9 — Splits at patient and site level, always.** Image-level splitting inflates AUROC 2–20+ points; the leakage audit is a failing CI test.
- **A10 — Calibration is infrastructure.** Cascades, conformal sets, deferral, and fusion are only correct on honest probabilities. Calibrate per model, after ensembling, per site; apply the analytic prior-correction logit shift after rebalanced training.
- **A11 — Boring baselines first, always as a control arm.** Tuned ERM + augmentation + weight averaging + group balancing beats most fancy robustness methods; DFR is the default subgroup fix.
- **A12 — Security is an integrity problem, not evasion.** Injection riding on clinical data (94% attack success documented), poisoning of the unlabeled pool (0.1% suffices), the loop's own writable stores. Only architectural controls hold.
- **A13 — Abstention is a first-class output.** Confidence from structural signals (vote entropy, disagreement, conformal set size, OOD distance), never verbalized confidence or raw softmax.
- **A14 — Design PCCP-shaped from day one.** Pre-specified modifications, validation methodology, impact assessment, monitoring, rollback.

## Part 2 · Harness technique catalog

Legend: ● proven · ◐ emerging · ○ speculative

**2A Spatial attention & active perception (12):** ● coarse-to-fine ROI zoom loop (depth 2–3, 8–16 crops) · ● multi-scale pyramid tiling / sliding window · ◐ saliency/surprise-guided region selection · ● 3D volume navigation as tools (`get_slice`, `reformat`, `MIP`, `set_window`, `measure_HU`) · ● set-of-mark overlays · ● visual prompt shapes (ellipse > rectangle > scribble; MedVP +12.2 pts) · ◐ coordinate-frame scaffolds · ● guided visual search (V*/SEAL) · ◐ attention injection (decoy controls for automation bias) · ● region-scoped sub-queries · ● multi-view/multi-slice assembly · ● task-routed reformats (MIP 8–10 mm solid nodules; MinIP ground-glass; CPR tubular; re-localize on source before reporting coordinates).

**2B Model orchestration (10):** ● detector-first grounding (nnU-Net/nnDetection/DETR propose; VLM verifies crops) · ● cheap-screener → specialist cascade (thresholds optimized jointly; 0.97³ ≈ 0.91) · ◐ heterogeneous ensembling, LLM as aggregator · ● modality/protocol router with capability guardrails · ◐ anomaly-map fusion (PatchCore, AnomalyCLIP) · ● SAM/MedSAM2 segmentation prompting (never autonomous zero-shot detection) · ● anatomical grounding & plausibility gating (TotalSegmentator) · ● spatial aggregation mechanics (weighted box fusion; STAPLE) · ● LLM-as-arbiter on disagreement · ● the architectural boundary: VLM routes/characterizes/communicates; CNN/ViT are the eyes; never a VLM-only negative below its resolution floor.

**2C Multi-agent & adversarial verification (9):** ◐ multi-agent debate (2–3 rounds, rebuttals cite new visual evidence, blind round 1) · ◐ role-conditioned expert panel · ● symmetric adversarial re-check (FP-hunter must name a specific alternative; FN-hunter re-searches blind spots; track verifier precision AND recall) · ● blinded double reading + arbitration · ◐ independent cross-family judge · ◐ grounded re-verification pass (crop every asserted box → fresh context, narrow yes/no; highest-yield anti-hallucination mechanic) · ◐ counterfactual dual-hypothesis reasoning · ● critique-and-revise loop (capped 2–3; no confidence change without a new tool call) · ◐ hallucination detectors (RadFlag, ReXTrust; RadGraph-F1/CheXbert/GREEN, never BLEU/ROUGE).

**2D Sampling & test-time compute (8):** ● self-consistency/majority vote (the disagreement rate is the most valuable output) · ◐ best-of-N with verifier reranking · ● harness-orchestrated TTA (invert transforms before aggregation; never flip laterality-sensitive images) · ● prompt ensembles & question decomposition · ◐ input-transform ensembles (union raises recall; intersection raises precision) · ◐ adaptive difficulty-gated compute · ○ MCTS over diagnostic actions · ◐ guarded test-time adaptation (BN-stats only; TENT collapses under screening imbalance; frozen shadow model + rollback).

**2E Prompting & guidance (15):** ● visual CoT: describe before judge · ● checklist-driven search (5–8-item groups) · ● controlled-lexicon slot filling (BI-RADS/Lung-RADS descriptors; code computes the category) · ● localization before classification · ● bilateral/contralateral comparison (identical preprocessing both sides) · ● temporal prior comparison (prior impression stripped during perception) · ◐ contrastive/counterfactual prompting · ● normal-anchoring · ● ranked differential with likelihood bands + abstention · ◐ two-stage blind-first protocol · ◐ cohort/prevalence conditioning · ● scale grounding (pixel spacing injected; sizes computed in code; flag >20% disagreement) · ● report-template conditioning (integration stage only) · ◐ reasoning-effort control (CoT is a per-task hyperparameter; documented −5.7% cases) · ◐ order/negation debiasing.

**2F Retrieval & knowledge (5):** ● kNN case atlas (class-stratified, distance-aware — no neighbors → abstain; grows monotonically) · ● guideline retrieval as deterministic tool (version-pinned signed corpus) · ● few-shot exemplars (GPT-4V 10-shot: 83.3% MHIST, 88.3% PCam; 40%→90% from 0→20-shot) · ◐ retrieval-augmented exemplar selection · ◐ hard-negative mimic exemplars.

**2G Output control & abstention (9):** ● grammar-constrained extraction (scratchpad turn first; `abstain`/`not_assessable` enums everywhere) · ◐ grounded reporting (every claim carries a box; deterministic checker) · ◐ evidence ledger (append-only; no confidence raise without a new tool call) · ● tool-result primacy (every numeric token post-validated) · ● conformal prediction sets (class-conditional Mondrian; marginal coverage is vacuous at 0.5% prevalence) · ● selective prediction / learning to defer (risk–coverage curve; audit deferral by subgroup) · ● operating-point control (never 0.5; per site, per generation) · ● image-quality gating preflight · ● structured human handoff.

**2H Preprocessing & image engineering (10):** ● DICOM correctness (MONOCHROME1, rescale, VOI LUT, PixelSpacing, laterality; ONE shared module — train/serve skew is the largest silent killer) · ● windowing sweep / multi-window stacking · ● CLAHE selectively (original alongside; confirm on original) · ● stain augmentation over normalization · ● artifact detection & routing · ● shortcut/spurious-cue suppression (OCR-redact burned-in text; serves de-ID AND injection defense) · ● super-resolution for display only · ● anatomy suppression: deterministic (invents nothing) vs generative (no finding unless re-identified on unmodified source; no negative on a generated image) · ● de-identification as a continuous process · ● anatomy masking as the conservative alternative.

**2I Registration & correspondence (6):** ● the correspondence ladder (exact → deformable+QC → landmark → coarse anatomical address → refuse and say so) · ● toolkit tiers (SimpleITK affine init first → ANTs SyN/ConvexAdam/deedsBCV → learned + instance optimization) · ● registration QC gate (Jacobian folding, inverse-consistency, per-organ Dice; TRE suppresses claims < ~2× predicted error; median and p95, never mean) · ● mammography is special (nipple-arc geometry, never pixel subtraction of two breasts) · ● lesion tracking (bipartite assignment with cost cap; propagation + independent de-novo detector; new lesion needs independent detection AND failed correspondence) · ● change quantification error floor (25% volume / 2 mm before growth arithmetic; "stable within measurement error" is a first-class outcome).

**2J Uncertainty, calibration & OOD (8):** ● deep ensembles M=5–10 (measure member error correlation) · ● MC dropout / SWAG / Laplace · ◐ evidential heads & rater-ambiguity (preserve per-rater annotations) · ◐ conformal risk control (FNR control on tumor masks) · ● the calibration stack (per model → after ensembling → per site → prevalence-corrected; adaptive-bin ECE + Brier, never binned ECE alone) · ◐ subgroup multicalibration · ● OOD three tiers: far-OOD reject / covariate shift recalibrate / near-OOD escalate to human, never reject · ● drift monitoring with SPC (classify shift type before any retraining).

**2K Consistency & explainability verification (9):** ● segmentation–classification agreement · ● multi-view/contralateral/longitudinal consistency · ● metamorphic invariance tests · ● ontological & physical plausibility rules · ◐ cycle consistency · ● attribution shortcut auditing (heatmap is a shortcut screen, never evidence of correctness) · ◐ counterfactual editing oracle (RoentMod pattern) · ◐ image-ablated control run (98.5% unchanged without the image documented; standing CI gate) · ◐ adversarial self-testing red team.

## Part 3 · Recursive self-improvement loop

**The lever ladder (cheap first):** scaffold/prompt optimization → exemplar-bank & retrieval growth → threshold/calibration refits → head retraining on frozen cached embeddings → LoRA/full retraining → RL. GEPA reportedly beats GRPO by 6–20% with 35× fewer rollouts.

**3A Truth channels (the only sources of new information):** human adjudication (blinded panels) · report-mined labels (LLM extraction + Dawid–Skene) · outcome harvesting (pathology, registry, 12–24-month interval outcomes) · shadow-mode deployment harvesting · proofread-not-annotate telemetry.

**3B Self-training:** ● pseudo-labeling with confidence gating; Noisy Student (student strictly disadvantaged) · ● Mean Teacher/UA-MT · ● FixMatch→FlexMatch per-class thresholds · ● cross-pseudo-supervision (CNN × transformer) · ● UPS · ◐ MedSAM mask amplification · ● MIL bootstrapping for WSIs · ● periodic SSL re-pretraining.

**3C Targeted acquisition:** ● active learning (entropy/BALD/BADGE/Core-Set) · ● hard-example mining (site- and subgroup-stratified; exclude short-follow-up "negatives") · ● dataset cartography · ◐ automated error-slice discovery (Domino/Spotlight) · ● disagreement mining · ● label-noise auditing (cleanlab; flag only, never auto-relabel gold).

**3D Weight-level:** ● distillation (amplify-then-distill) · ◐ LoRA adapter library with routing · ● RAFT and DPO on verified pairs · ◐ RLVR/GRPO on verifiable rewards; process reward models · ◐ STaR rationale bootstrapping · ● model soups / TIES-DARE merging · ● nnU-Net/Auto3DSeg self-configuring retraining · ● continual learning (replay, EWC, backward-transfer matrix, BCWI) · ● federated/swarm loops.

**3E Scaffold-level (cheapest — build first):** ◐ GEPA/MIPROv2/TextGrad prompt compilation (holdout the optimizer never sees) · ◐ exemplar-bank growth · ● per-generation refits (thresholds, calibration, ensemble weights, cascade cutoffs) · ◐ Reflexion episodic memory, guarded (typed records never rendered as instructions; probationary → confirmed; TTL; retrieval-blind shadow arm) · ○ ADAS/DGM scaffold search, contained (highest-risk item) · ◐ tool-selection/escalation-policy RL.

**3F Synthetic data:** ◐ class-conditional diffusion; mask-conditioned generation · ● virtual imaging trials (VICTRE) · ◐ counterfactual inpainting pairs; generative replay · ● mandatory guards: memorization screening (37.2% documented), parent-study provenance, real-vs-synthetic discriminator gate, hard exclusion from every eval set.

**3G Loop safeguards — the ratchet (ALL mandatory):** sealed eval tiers (hash-locked, query-budgeted, aggregate-only, access-logged) · tiered label provenance (gold/silver/bronze/synthetic; no machine label promoted to gold; stratified human audit) · real-data anchoring (minimum real fraction at the batch sampler) · collapse early-warning monitors (pseudo-label entropy, feature effective rank, ensemble disagreement, rarest-decile recall) · auto-generated regression suite (every confirmed error a locked test; negative-flip gate) · non-inferiority promotion gates per subgroup with worst-slice floor · leakage & contamination CI audit · independent rotated cross-family verifiers + one frozen audit judge · blinded reader control arm (5–10%) · shadow → canary → champion (promotion = registry pointer move; rehearsed rollback of every mutable store) · label-free performance estimation (CBPE) + outcome reconciliation · PCCP governance: the LLM never holds write access to gate rules, test manifests, splits, de-identification, allowlists, or the promotion path.

## Part 4 · Subgroup robustness & shift

Central replicated finding: under equal tuning budgets, GroupDRO/IRM/JTT/DANN do NOT reliably beat group-balanced reweighting or well-tuned ERM. Every remediation A/B includes a tuned-ERM control arm.

Free wins first: worst-group model selection and early stopping · SWAD/soups · SAM optimizer · site-stratified minibatches · heavy augmentation. **DFR is the default remediation.** The one-probe diagnostic gates everything (linear head on frozen features from a balanced slice sample: near-parity → head fix; else representation fix — never naive full fine-tuning). Escalation ladder: <50–100 positives → generative augmentation; continuous nuisance → HSIC; label-correlated → conditional adversaries; determined label → representation intervention; annotation bias → relabel. Screening-imbalance resolution always on: train rebalanced, restore prior with analytic logit shift, report both views. Precondition tests before any group objective. Slice-discovery → group-label bridge (GEORGE/EIIL) on five disjoint splits. FM caveat: pathology FMs encode site more than biology. Statistical floor: Camelyon17 ERM seed variance spans ~65–93%; multi-seed distributions required.

## Part 5 · Security & integrity

Injection surfaces: DICOM free-text tags (allowlist) · burned-in pixel text (multi-contrast OCR gate; sub-visual 6-pt prompts flip 33–67% of diagnoses) · PDFs/referrals · filenames · MCP tool descriptions (pin + rug-pull detection) · retrieved guidelines (closed signed corpus).

Architectural controls (the only ones that hold): provenance-typed context envelopes · quarantined-LLM ingestion (schema-constrained, closed-enum outputs) · plan-then-execute control-flow integrity · deny-by-default tool allowlists enforced outside the model · deterministic output schemas as the firewall.

Poisoning: budgets tiny (0.1%; ~250 documents); provenance beats detection (content-addressed append-only manifest, revoke-by-source rebuild); randomized preprocessing jitter; slice-conditioned canary metrics. Model-authored stores: typed imperative-free records, human-gated outcome-anchored promotion, TTL/dedup/pruning, hash-chained audit log, dual control. Supply chain: safetensors-only, `weights_only=True`, no `trust_remote_code`, SHA pinning, sigstore + behavioral acceptance, SBOM. PHI: boundary decision before pipeline code; Safe Harbor = all 18 identifiers; traces/logs/caches are a PHI store; embeddings and synthetic images are re-identifying. CI security gates: injection ASR + clinical-harm rate next to AUROC; security telemetry monitored but EXCLUDED from the optimization objective. Written threat model mapped to MITRE ATLAS / NIST AI 100-2; an immutable core the self-modifying component provably cannot write.

## Part 6 · Evaluation doctrine

Metrics: AUROC for selection; sensitivity at fixed specificity (screening ~96–98%) for release; FROC/CPM with versioned hit criterion; lesion-wise Dice+NSD+HD95; QWK for grading (human κ 0.6–0.8 is the ceiling — beating it means leakage); time-dependent AUC/C-index; adaptive-bin ECE + Brier; risk–coverage curve as primary deployment metric; clinical endpoints (CDR/1000, recall rate, interval-cancer rate, PPV1/PPV3).

Statistics: patient-level clustered bootstrap; DeLong/McNemar paired tests; multiplicity control; multi-seed distributions; Metrics Reloaded; MRMC for reader studies. Splits: patient- and site-preserved; temporal holdout → leave-one-site-out → external cohort; enriched-set PPV is meaningless.

Standing CI gates: leakage audit · image-dependence floor · consistency/reversal rates · per-class sensitivity floors · small-lesion sensitivity by diameter · robustness suite · injection ASR · subgroup non-inferiority · negative-flip rate · per-slice calibration · deferral-rate non-regression · deterministic double-run agreement. Reporting: CLAIM 2024 / TRIPOD+AI as machine-checkable specs.

## Part 7 · Domain grounding

Starter datasets: **RSNA Screening Mammography** (default: outcome labels, two sites, ~2% prevalence) · CAMELYON16/17 (five centers) · PANDA (ordinal κ) · LIDC-IDRI/LUNA16 (FROC, per-rater) · ISIC (dedup mandatory; DDI/Fitzpatrick17k fairness) · MedMNIST (smoke tests).

Roster: nnU-Net v2 ResEnc, MONAI Auto3DSeg, nnDetection, TotalSegmentator · pathology FMs UNI2/Virchow2/Prov-GigaPath/CONCH (+TITAN/PRISM; TRIDENT/CLAM cached embeddings) · MedSAM/MedSAM2/VISTA3D · VLM orchestration layer: MedGemma 1.5, MedSigLIP, MAIRA-2, CheXagent, LLaVA-Med, BiomedCLIP; comparator floors Sybil, Mirai · registration SimpleITK/ANTs/ConvexAdam/uniGradICON · infra FastAPI→Triton, vLLM, MLflow, DVC/lakeFS, Label Studio/MONAI Label/QuPath, cleanlab, NannyML.

Regulatory: FDA PCCP (final Dec 2024, broadened Aug 2025) authorizes pre-specified bounded verified change cycles with human sign-off and rollback — not field online learning. EU MDR class IIa+ with AI Act Art. 6(1) from Aug 2027. Post-market drift monitoring is a legal obligation and the technical prerequisite of the loop.

## Part 8 · Build plan

v0 scope: one task, one public dataset, one metric; solo-dev scale (24–32 GB GPU);
frozen-FM encoder + light head + Claude Agent SDK harness; local-first for pixels.

### Phase 0 — Scoping
- **T-0.1** Choose modality, task, dataset (default RSNA Mammography or CAMELYON17); fix primary metric (sens @ 96% spec + partial AUC) and operating context. ✅
- **T-0.2** One-page threat model + PHI boundary decision. ✅ `THREAT_MODEL.md`
- **T-0.3** Repo scaffold (`src/data`, `src/models`, `src/eval`, `src/serving`, `src/harness/tools`, `gates/`, `tests/`); pinned environment. ✅

### Phase 1 — Data pipeline & locked test set (before any model)
- **T-1.1** Ingest → canonicalize (DICOM correctness; golden-fixture tests) → de-identify. ✅ `data/dicom_canonical.py`, `data/deid.py`
- **T-1.2** Patient- and site-grouped splits persisted as versioned files; CI leakage tests. ✅ `data/splits.py`, `eval/leakage.py`
- **T-1.3** Hash-seal the test set; scoring service with access accounting and query budget; disjoint calibration/threshold/slice-discovery splits. ✅ `eval/sealed.py`
- **T-1.4** Metric implementations validated against known cases; versioned FROC hit criterion. ✅ `eval/metrics.py` (FROC: Phase 7, detection-only v1 per D4)

### Phase 2 — Baselines
- **T-2.1** Frozen FM encoder + linear/ABMIL head on cached embeddings. ✅ scaffolded (`models/features.py`, `models/head.py`; real FM swap-in pending)
- **T-2.2** Deep ensemble (5 seeds) + temperature calibration + prevalence correction; multi-seed distribution. ◻ partial (calibration + prior shift shipped)
- **T-2.3** MLflow registry + model card; FastAPI wrapper importing the same preprocessing module. ◻ partial (`serving/app.py`; MLflow pending)

### Phase 3 — Eval gate service
- **T-3.1** Callable PASS/FAIL gate: paired non-inferiority, subgroup floors, calibration, negative-flip, determinism double-run. ✅ `eval/gate.py` (small-lesion & image-dependence checks pending real data)
- **T-3.2** Regression suite + auto-ingesting failure bank; synthetic-phantom tests of the plumbing. ✅ `data/phantom.py`, `tests/`
- **T-3.3** Gate rules in a protected path the harness's service account cannot write. ✅ `gates/` + CI immutability test

### Phase 4 — The harness
- **T-4.1** Full toolbelt: `describe_store`, `run_detector` (primary + blindspot profiles), `crop_region`, `segment`, `measure`, `compare_prior` (registration + QC inside the tool; refuses on failed correspondence), `retrieve_similar`, `lookup_criteria`, `run_eval_gate`, `submit_review`; handle-passing artifact store; Claude Agent SDK in-process MCP server with built-ins stripped; deny-by-default allowlist enforced in our code (SDK PreToolUse hooks are a belt-and-suspenders add-on, pending SDK API stabilization). ✅
- **T-4.2** Deterministic outer state machine (ingest → preflight QC → detect → TTA verify → zoom re-verify → FP/FN hunt → aggregate → adjudicate → conformal policy → report); LLM only at the adjudication node. ✅ `harness/state_machine.py`
- **T-4.3** Inference stack v1: detector-first grounding + TTA self-consistency + zoom crop-and-re-detect (A1 clean-room look) + symmetric FP-hunter (named-alternative vetoes) / FN-hunter (second detector family over unexamined regions) + Mondrian conformal deferral + structured extraction with abstention. ✅ (cross-*model-family* verification upgrades with real specialist models)
- **T-4.4** Evidence ledger (hash-chained) ✅; image-ablated control wired into CI ✅ (`test_image_ablated_control`); OTel/MLflow trace export pending (with real-data phase).
- **T-4.5** Ablation runner: harness vs detector-alone on a shared case set (`eval/ablation.py`); VLM-alone arm joins when an LLM adjudicator is attached. ✅

### Phase 5 — The flywheel
- **T-5.1** Review queue (AL score × disagreement × cleanlab × novelty) → Label Studio; typed feedback schema; ~20% random audit. ◻
- **T-5.2** Silver-tier pseudo-labeling with per-class adaptive thresholds, provenance manifest, stratified audit with precision floor and auto-freeze. ◻
- **T-5.3** Nightly candidate generation on cached embeddings → gate → human-approved promotion as registry alias move; max one promotion/week. ◻
- **T-5.4** Exemplar bank + kNN atlas wired in (class-stratified, distance-gated). ◻ partial (`retrieve_similar` reads the atlas)
- **T-5.5** Prompt optimization (GEPA/MIPROv2) on dev traces with a never-seen holdout; prompt registry with versioned A/B. ◻
- **T-5.6** Collapse/diversity monitors, backward-transfer matrix, generation lineage with rehearsed rollback. ◻

### Phase 6 — Hardening
- **T-6.1** Drift monitoring (embedding SPC, CBPE, shift-type classifier) with actions wired. ◻
- **T-6.2** Security gates in CI (injection corpus incl. poisoned DICOM tags and sub-visual pixel prompts; ASR threshold); revoke-by-source rebuild rehearsal. ◻
- **T-6.3** Subgroup remediation ladder (one-probe → DFR → escalations) driven by slice discovery. ◻ partial (`dfr_refit` shipped)
- **T-6.4** Shadow/canary machinery; PCCP-shaped change-log generator. ◻

### Phase 7 — Extensions (only if warranted)
- **T-7.x** Second modality · multi-agent debate for the hard tail · registration-based prior comparison · Triton serving · federated training · RLVR/DPO fine-tuning.

> **Explicitly deferred:** Kubernetes, feature stores, custom architecture research,
> multi-modality breadth before Phase 7. Build the measurement apparatus before the
> thing being measured.

## Part 9 · Pitfall register (check before every phase)

Image-level splits (2–20+ fake AUROC points) · PPV from an enriched test set · threshold chosen on the test set · MONOCHROME1/rescale bugs · WSI magnification assumed not read from MPP · preprocessing statistics fit before splitting · BI-RADS treated as pathology truth · interval cancers ignored · case-level AUROC for localization · FM-pretraining contamination as "external validation" · adaptive overfitting of a repeatedly-queried holdout · pseudo-labels leaking into eval · verifier silently suppressing true positives · hedging drift in critique loops · false consensus in agent panels · confirmation cascades · hallucinated coordinates and mm values · image-blind shortcutting · prior-report impressions copied forward · CLAHE halos read as findings · flip-TTA laterality errors · sycophancy toward injected heatmaps · cascade sensitivity multiplying down · marginal conformal coverage hiding the malignant class · TENT collapse · "ensembles" of one backbone · deferral dumping minority cases on humans · unregistered VLM comparison of priors · volume doubling below the error floor · MIP for ground-glass · greedy NN lesion matching · generative suppression erasing a real lesion · rebalancing without prior correction (50–200×) · DANN destroying label-correlated site signal · CutMix occluding the only lesion · mined "FPs" that are unreported cancers · seed variance dwarfing the claimed gain · rubber-stamp oversight (400 labels/hour) · pickle checkpoints · mutable model tags · trace logs as an unaudited PHI store · the agent holding write access to its own gates.

## Part 10 · Open decisions — resolved at kickoff (2026-08-29)

| # | Decision | Resolution |
|---|---|---|
| D1 | Task & dataset | RSNA Screening Mammography shape; phantom dataset in-repo until data lands |
| D2 | Compute | Apple Silicon laptop, no CUDA GPU → frozen-embedding workflows; heavy training deferred |
| D3 | Orchestrator hosting | Hosted Claude (text + handles only); revisit if data ceases to be public |
| D4 | v1 ambition | Detection-only; FROC localization deferred to Phase 7 |
