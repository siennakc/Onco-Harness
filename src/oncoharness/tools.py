"""Deterministic tools (T-4.1). The only place work happens.

The LLM plans and adjudicates; these tools perceive, measure, and retrieve.
Every call writes a ``tool_call`` + ``tool_result`` pair into the evidence
ledger, and every numeric fact downstream must cite one of those entries.

Registry is deny-by-default: only tools registered here are callable, whether
invoked by the deterministic state machine or exposed to the LLM over MCP.

Every call is also metered (U1: E1): the belt times the tool body and counts
detector pixels into the case's :class:`~oncoharness.telemetry.CostMeter`.
The meter is code-side only — deliberately not a registered tool (S8).

With a :class:`~oncoharness.memo.ToolMemo` attached (U3: E6, opt-in), calls
to deterministic tools are served from the content-hash cache when the same
bytes and parameters were already computed: the call is still audited (a
``tool_call`` plus a ``tool_result_cached`` entry citing the original
evidence) and still metered (as a cached hit), but the tool body never runs
— which is what makes determinism double-runs and nightly replays nearly
free. Only this belt reads or writes the memo; it is never a tool (S8).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .reference.detector import DoGBlobDetector
from .reference.features import embed_crop
from .ledger import EvidenceLedger
from .memo import ToolMemo
from .store import ArtifactStore
from .telemetry import CostMeter


class Toolbelt:
    """All tools bound to one case-processing context (store + ledger)."""

    def __init__(
        self,
        store: ArtifactStore,
        ledger: EvidenceLedger,
        atlas_path: str | Path | None = None,
        criteria_path: str | Path | None = None,
        detector: DoGBlobDetector | None = None,
        gate_rules_path: str | Path = "gates/gate_rules.yaml",
        meter: CostMeter | None = None,
        gate_audit_path: str | Path = "runs/gate_audit.jsonl",
        memo: ToolMemo | None = None,
    ) -> None:
        self.store = store
        self.ledger = ledger
        self.meter = meter if meter is not None else CostMeter()
        self.memo = memo
        self.detector = detector or DoGBlobDetector()
        # A second, differently-parameterized family for blind-spot re-search
        # (axiom A5): tuned to a different scale band and a lower floor.
        self.detector_profiles: dict[str, DoGBlobDetector] = {
            "primary": self.detector,
            "blindspot": DoGBlobDetector(
                sigma_small=3.0, sigma_large=9.0, score_threshold=0.30
            ),
        }
        self.atlas_path = Path(atlas_path) if atlas_path else None
        self.criteria_path = Path(criteria_path) if criteria_path else None
        self.gate_rules_path = Path(gate_rules_path)
        self.gate_audit_path = Path(gate_audit_path)
        self._registry: dict[str, Callable[..., dict]] = {
            "describe_store": self.describe_store,
            "run_detector": self.run_detector,
            "crop_region": self.crop_region,
            "segment": self.segment,
            "measure": self.measure,
            "compare_prior": self.compare_prior,
            "retrieve_similar": self.retrieve_similar,
            "lookup_criteria": self.lookup_criteria,
            "run_eval_gate": self.run_eval_gate,
            "submit_review": self.submit_review,
        }

    # -- plumbing --------------------------------------------------------
    def call(self, tool_name: str, **kwargs: Any) -> dict:
        """Single audited entry point. Unknown tools are refused loudly."""
        if tool_name not in self._registry:
            raise PermissionError(f"tool {tool_name!r} is not in the allowlist")
        memo_key = (
            self.memo.key(tool_name, kwargs, self.store) if self.memo is not None else None
        )
        call_ref = self.ledger.append("tool_call", {"tool": tool_name, "args": _safe(kwargs)})
        if memo_key is not None:
            t0 = time.perf_counter()
            cached = self.memo.get(memo_key)
            if cached is not None:
                # U3 memo hit: audited and metered like any call (so the U1
                # invariant holds and cached_hits counts it), but the tool
                # body never runs. The cached entry cites the original
                # result's ledger ref, so replays and the image-ablated
                # control can tell cached evidence from fresh (E6/S8).
                self.meter.record_tool(
                    tool_name, (time.perf_counter() - t0) * 1000.0, pixels=0, cached=True
                )
                result = cached["result"]
                cached_ref = self.ledger.append(
                    "tool_result_cached",
                    {
                        "tool": tool_name,
                        "call_ref": call_ref,
                        "key": memo_key,
                        "original_ref": cached["original_ref"],
                    },
                )
                result["evidence_ref"] = cached_ref
                return result
        t0 = time.perf_counter()
        ok = False
        try:
            result = self._registry[tool_name](**kwargs)
            ok = True
        finally:
            # Metered even on failure, so cost.tool_calls always equals the
            # ledger's tool_call entries for the case (U1 invariant).
            self.meter.record_tool(
                tool_name,
                (time.perf_counter() - t0) * 1000.0,
                pixels=self._detector_pixels(tool_name, kwargs) if ok else 0,
            )
        result_ref = self.ledger.append(
            "tool_result", {"tool": tool_name, "call_ref": call_ref, "result": _safe(result)}
        )
        result["evidence_ref"] = result_ref
        if memo_key is not None:
            self.memo.put(memo_key, result)
        return result

    def _detector_pixels(self, tool_name: str, kwargs: dict) -> int:
        """Detector input pixels (E1): h*w from the manifest, no pixel load."""
        if tool_name != "run_detector":
            return 0
        try:
            shape = self.store.info(kwargs["image_handle"]).shape
        except Exception:
            return 0
        pixels = 1
        for dim in shape:
            pixels *= int(dim)
        return pixels

    # -- tools -----------------------------------------------------------
    def describe_store(self) -> dict:
        return {"artifacts": self.store.describe()}

    def run_detector(self, image_handle: str, profile: str = "primary") -> dict:
        if profile not in self.detector_profiles:
            raise ValueError(f"unknown detector profile {profile!r}")
        pixels = self.store.get(image_handle)
        candidates = self.detector_profiles[profile].propose(pixels)
        return {
            "image_handle": image_handle,
            "profile": profile,
            "candidates": [
                {"candidate_id": f"{profile[0]}{i}", "box": list(c.box), "score": c.score}
                for i, c in enumerate(candidates)
            ],
        }

    def crop_region(self, image_handle: str, box: list[int]) -> dict:
        pixels = self.store.get(image_handle)
        x0, y0, x1, y1 = (int(v) for v in box)
        h, w = pixels.shape
        x0, x1 = max(0, x0), min(w, x1 + 1)
        y0, y1 = max(0, y0), min(h, y1 + 1)
        crop = pixels[y0:y1, x0:x1]
        info = self.store.put(crop, kind="crop", meta={"source": image_handle, "box": [x0, y0, x1, y1]})
        return {"crop_handle": info.handle, "shape": list(crop.shape)}

    def measure(self, image_handle: str, box: list[int], pixel_spacing_mm: list[float]) -> dict:
        """All sizes in mm are computed here, never authored by the LLM."""
        x0, y0, x1, y1 = (int(v) for v in box)
        sy, sx = float(pixel_spacing_mm[0]), float(pixel_spacing_mm[1])
        width_mm = abs(x1 - x0) * sx
        height_mm = abs(y1 - y0) * sy
        pixels = self.store.get(image_handle)
        region = pixels[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1]
        return {
            "width_mm": round(width_mm, 2),
            "height_mm": round(height_mm, 2),
            "long_axis_mm": round(max(width_mm, height_mm), 2),
            "mean_intensity": round(float(region.mean()) if region.size else 0.0, 4),
            "contrast": round(
                float(region.mean() - pixels.mean()) if region.size else 0.0, 4
            ),
        }

    def segment(
        self, image_handle: str, box: list[int], pixel_spacing_mm: list[float] | None = None
    ) -> dict:
        """Grow a connected component from the box center; quantify its shape.

        Deterministic descriptors (area, equivalent diameter, circularity)
        computed from the mask — the conservative anti-hallucination check:
        a "mass" with no coherent component or ridge-like geometry earns a
        named alternative for the FP-hunter.
        """
        pixels = self.store.get(image_handle)
        sy, sx = (pixel_spacing_mm or [0.1, 0.1])
        x0, y0, x1, y1 = (int(v) for v in box)
        h, w = pixels.shape
        x0, x1 = max(0, x0), min(w - 1, x1)
        y0, y1 = max(0, y0), min(h - 1, y1)
        crop = pixels[y0 : y1 + 1, x0 : x1 + 1]
        if crop.size < 9:
            return {"found": False, "reason": "box too small"}

        ch, cw = crop.shape
        center = crop[ch // 4 : 3 * ch // 4, cw // 4 : 3 * cw // 4]
        ring_mean = (crop.sum() - center.sum()) / max(crop.size - center.size, 1)
        threshold = 0.5 * (float(center.mean()) + float(ring_mean))
        mask = crop >= threshold

        # Seed at the brightest pixel of the central region, not the geometric
        # center: a detector box clipped at the image edge shifts the crop
        # center off the structure it was proposed for.
        cy_off, cx_off = np.unravel_index(int(center.argmax()), center.shape)
        seed = (cy_off + ch // 4, cx_off + cw // 4)
        if not mask[seed]:
            return {"found": False, "reason": "no coherent component near box center"}
        # BFS flood fill of the seed's connected component (4-neighborhood).
        component = np.zeros_like(mask)
        stack = [seed]
        component[seed] = True
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < ch and 0 <= nx < cw and mask[ny, nx] and not component[ny, nx]:
                    component[ny, nx] = True
                    stack.append((ny, nx))

        area_px = int(component.sum())
        padded = np.pad(component, 1)
        boundary = component & ~(
            padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        )
        perimeter_px = max(int(boundary.sum()), 1)
        circularity = min(1.0, 4 * np.pi * area_px / (perimeter_px**2))
        rows = np.flatnonzero(component.any(axis=1))
        cols = np.flatnonzero(component.any(axis=0))
        height = int(rows[-1] - rows[0] + 1)
        width = int(cols[-1] - cols[0] + 1)
        aspect_ratio = max(height, width) / max(min(height, width), 1)
        area_mm2 = area_px * sy * sx
        info = self.store.put(
            component.astype(np.float32), kind="mask",
            meta={"source": image_handle, "box": [x0, y0, x1, y1]},
        )
        return {
            "found": True,
            "mask_handle": info.handle,
            "area_mm2": round(float(area_mm2), 2),
            "equivalent_diameter_mm": round(float(2 * np.sqrt(area_mm2 / np.pi)), 2),
            "circularity": round(float(circularity), 3),
            "aspect_ratio": round(float(aspect_ratio), 2),
            "mask_fraction_of_box": round(area_px / crop.size, 3),
        }

    def compare_prior(
        self,
        current_handle: str,
        prior_handle: str,
        box: list[int],
        change_floor: float = 0.05,
    ) -> dict:
        """Prior comparison with registration + QC inside the tool (2I).

        Refuses the comparison outright when registration QC fails — the
        correspondence ladder ends at "refuse and say so", never at a guess.
        """
        from .reference.registration import apply_shift, register_translation

        current = self.store.get(current_handle)
        prior = self.store.get(prior_handle)
        reg = register_translation(current, prior)
        qc = {
            "shift_px": list(reg.shift),
            "ncc_before": round(reg.ncc_before, 4),
            "ncc_after": round(reg.ncc_after, 4),
            "passed": reg.passed_qc,
            "reason": reg.reason,
        }
        if not reg.passed_qc:
            return {
                "status": "no_valid_correspondence",
                "qc": qc,
                "note": "comparison refused: no usable registration between studies",
            }
        aligned = apply_shift(prior, reg.shift)
        x0, y0, x1, y1 = (int(v) for v in box)
        cur_region = current[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1]
        pri_region = aligned[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1]
        delta = float(cur_region.mean() - pri_region.mean()) if cur_region.size else 0.0
        if abs(delta) < change_floor:
            change = "stable_within_measurement_error"
        else:
            change = "increased" if delta > 0 else "decreased"
        return {
            "status": "compared",
            "qc": qc,
            "region_delta_intensity": round(delta, 4),
            "change_floor": change_floor,
            "change": change,
        }

    def retrieve_similar(self, crop_handle: str, k: int = 5) -> dict:
        """kNN case atlas (2F): distance-aware — no neighbors means abstain."""
        if self.atlas_path is None or not self.atlas_path.exists():
            return {"neighbors": [], "note": "atlas empty — treat as no-neighbor abstain signal"}
        atlas = json.loads(self.atlas_path.read_text())
        q = embed_crop(self.store.get(crop_handle))
        scored = []
        for entry in atlas["entries"]:
            d = float(np.linalg.norm(q - np.array(entry["embedding"])))
            scored.append({"case_id": entry["case_id"], "label": entry["label"], "distance": round(d, 4)})
        scored.sort(key=lambda e: e["distance"])
        return {"neighbors": scored[:k]}

    def lookup_criteria(self, topic: str) -> dict:
        """Guidelines applied from a version-pinned local corpus, never weights."""
        if self.criteria_path is None or not self.criteria_path.exists():
            return {"topic": topic, "entries": [], "corpus_version": None}
        corpus = json.loads(self.criteria_path.read_text())
        entries = [e for e in corpus.get("entries", []) if topic.lower() in e["topic"].lower()]
        return {"topic": topic, "entries": entries, "corpus_version": corpus.get("version")}

    def run_eval_gate(self, results_path: str) -> dict:
        """Run the conjunctive promotion gate on a results file.

        Read-only over ``gates/`` (the harness can execute the gate, never
        edit its rules). The results file is produced by a batch evaluation
        run, so no score in it was authored by a model's text output.

        Optional keys in the results file: ``cand_costs``/``champ_costs``
        (per-case cost dicts, U8 efficiency floor) and ``policy_id``. When
        a ``policy_id`` is present, the evaluation is counted in the
        hash-chained proposal audit (S3 accounting) — the caller triggers
        that entry but never authors it: its contents come solely from the
        gate result computed here.
        """
        from .gate import load_rules, run_gate
        from .sequential import CaseCost, ProposalAudit

        data = json.loads(Path(results_path).read_text())
        rules = load_rules(self.gate_rules_path)
        result = run_gate(
            rules,
            np.asarray(data["y_true"]),
            np.asarray(data["candidate_scores"]),
            np.asarray(data["champion_scores"]) if data.get("champion_scores") else None,
            data["patient_ids"],
            subgroups=data.get("subgroups"),
            candidate_scores_rerun=(
                np.asarray(data["candidate_scores_rerun"])
                if data.get("candidate_scores_rerun")
                else None
            ),
            cand_costs=(
                [CaseCost(**c) for c in data["cand_costs"]] if data.get("cand_costs") else None
            ),
            champ_costs=(
                [CaseCost(**c) for c in data["champ_costs"]] if data.get("champ_costs") else None
            ),
        )
        if data.get("policy_id"):
            ProposalAudit(self.gate_audit_path).log(
                policy_id=str(data["policy_id"]),
                gate_passed=result.passed,
                e_value=None,
                checks={c.name: c.passed for c in result.checks},
            )
        return {"passed": result.passed, "summary": result.summary()}

    def submit_review(self, case_id: str, reason: str, ranked_regions: list[dict]) -> dict:
        """Structured human handoff (2G): regions + measurements, not a bare flag."""
        self.ledger.append(
            "decision",
            {"action": "defer_to_human", "case_id": case_id, "reason": reason,
             "ranked_regions": ranked_regions},
        )
        return {"queued": True, "case_id": case_id}


def _safe(obj: Any) -> Any:
    """Ledger payloads are JSON: arrays and numpy scalars are summarized, not embedded."""
    if isinstance(obj, np.ndarray):
        return {"__array__": list(obj.shape)}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    return obj
