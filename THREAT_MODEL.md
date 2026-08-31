# Oncoscope Threat Model (T-0.2)

One page. Threat frame: **integrity, not evasion** (tasksheet Part 5). The adversary is
untrusted text riding in on clinical data, and the loop writing its own errors into
persistent stores — not a lab attacker crafting adversarial pixels.

## PHI boundary decision

**v0 uses public, de-identified datasets only** (RSNA Screening Mammography / phantoms).

- Pixels stay **local**: images are loaded, preprocessed, and analyzed on the local
  machine. The artifact store is on local disk.
- A **hosted LLM (Anthropic API) is acceptable for orchestration** because it receives
  *text and opaque handles only* — never pixels, never DICOM headers, never free-text
  fields from the source data (axiom A3 + injection surface control).
- If the project ever ingests non-public data, this decision is re-made first:
  either a BAA + verified zero-retention endpoint, or fully local orchestration
  (vLLM + MedGemma). That is a governance event, not a config change.

## Assets

1. Sealed evaluation sets and their manifests (`data/splits/`, hash-sealed).
2. Gate rules (`gates/`) and promotion machinery.
3. Label stores by provenance tier (gold / silver / bronze / synthetic).
4. Model-authored stores: exemplar bank, kNN atlas, prompt registry.
5. The evidence ledger (append-only, hash-chained).

## Threats → controls

| Threat | Control |
|---|---|
| Prompt injection via DICOM free-text tags | Tag allowlist in `deid.py`; the LLM never receives raw header text |
| Injection via burned-in pixel text | OCR-redaction gate before the VLM sees pixels (Phase 6; stub in preflight QC) |
| Poisoning of the unlabeled pool (0.1% suffices) | Content-addressed provenance manifest; tiered labels; no machine label promoted to gold |
| Loop writes to its own gates | `gates/` is a protected path; harness service account has no write access; CI test asserts no code writes there |
| Adaptive overfitting of the sealed test set | Query budget + access accounting in `eval/sealed.py`; aggregate-only responses |
| Verifier drift suppressing true positives | Symmetric FP/FN verification (axiom A5); verifier precision and recall tracked separately |
| Supply chain (pickled checkpoints) | safetensors-only policy; SHA-pinned model references; no `trust_remote_code` |
| Trace logs as a PHI store | v0 public data only; ledger content is text + handles + hashes, never pixels |

## Immutable core (the agent can never write)

Eval harness, sealed-set loader, de-identification, tool allowlists, provenance
writer, promotion predicates, `gates/`. Enforced by filesystem permissions in
deployment and by CI assertion in development.
