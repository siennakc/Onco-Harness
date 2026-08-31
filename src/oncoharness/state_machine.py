"""Deterministic outer state machine (T-4.2, T-4.3).

ingest -> preflight QC -> detect -> TTA verify -> zoom re-verify ->
FP-hunt / FN-hunt -> aggregate -> adjudicate -> conformal policy -> report

Control flow is code, never the model. The LLM sits only at the *adjudicate*
decision node, behind the ``Adjudicator`` protocol; a rule-based adjudicator
ships as the default so the entire pipeline runs, tests, and gates without an
LLM in the loop (and serves as the control arm in T-4.5 ablations).

Inference stack v1 (T-4.3):
- TTA self-consistency: label-preserving translations only (never laterality
  flips), inverted before aggregation, IoU-clustered; findings must reproduce
  in >= k of N reads; the disagreement rate is a first-class output.
- Zoom re-verification (axiom A1): every kept candidate is cropped from the
  original full-resolution buffer and re-detected in a clean-room look; a
  finding that does not survive its own crop is dropped.
- Symmetric verification (axiom A5): the FP-hunter may veto a candidate only
  by naming a specific alternative (from segmentation geometry); the
  FN-hunter re-searches with a second detector family in regions the primary
  never proposed, and an unexplained blind-spot hit blocks a clean negative.
- Conformal deferral (axiom A13): an ambiguous or empty Mondrian prediction
  set overrides a confident-sounding decision with deferral.
- Cost telemetry (U1: E1): ``run_case`` brackets the toolbelt's CostMeter, so
  every report carries the case's cost block — tool calls, pixels, wall time,
  LLM spend — into the ledger's ``claim`` entry.
- Difficulty-gated lanes (U2: E3/E4/E5), strictly opt-in: with a
  :class:`~oncoharness.router.DifficultyRouter` attached, an obvious negative
  (no primary candidate above the lane threshold, blindspot clean) skips
  TTA/zoom/FP-hunt outright; stages early-exit when they can no longer
  change the decision; and the escalation adjudicator (the LLM) is invoked
  only to rescue deferrals — through the router's circuit breaker, after the
  distilled student (U7) had its shot. Without a router the pipeline runs
  exactly the fixed sequence above; the conformal safety net applies to
  every lane either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .conformal import MondrianConformal
from .reference.features import embed_crop
from .reference.head import LogisticHead
from .router import DifficultyRouter, Lane, RouteRecord
from .schemas import (
    Adjudication,
    Assessment,
    CaseDecision,
    CaseReport,
    Finding,
    QCVerdict,
)
from .tools import Toolbelt


class Adjudicator(Protocol):
    def adjudicate(self, request: dict) -> Adjudication: ...


@dataclass
class RuleBasedAdjudicator:
    """Deterministic adjudication: thresholds + deferral band + abstention.

    Also the "detector-alone" arm of the T-4.5 ablation.
    """

    recall_threshold: float = 0.65
    deferral_band: tuple[float, float] = (0.35, 0.65)
    max_disagreement: float = 0.4

    def adjudicate(self, request: dict) -> Adjudication:
        score = float(request["consistency"]["case_score"])
        disagreement = float(request["consistency"]["disagreement_rate"])
        blindspot = request.get("blindspot_candidates", [])
        per_candidate: dict[str, Assessment] = {}
        for cand in request["candidates"]:
            if cand.get("veto_reason"):
                per_candidate[cand["candidate_id"]] = Assessment.absent
            elif cand["reproduced_fraction"] >= 0.6 and cand.get("zoom_confirmed", True):
                per_candidate[cand["candidate_id"]] = (
                    Assessment.present if cand["score"] >= 0.5 else Assessment.uncertain
                )
            else:
                per_candidate[cand["candidate_id"]] = Assessment.absent

        if request["qc"] == QCVerdict.inadequate_defer.value:
            decision = CaseDecision.defer_to_human
            rationale = "image quality inadequate for assessment"
        elif disagreement > self.max_disagreement:
            decision = CaseDecision.defer_to_human
            rationale = f"self-consistency disagreement {disagreement:.2f} above band"
        elif self.deferral_band[0] <= score <= self.deferral_band[1]:
            decision = CaseDecision.defer_to_human
            rationale = f"score {score:.2f} inside deferral band"
        elif score > self.deferral_band[1]:
            decision = CaseDecision.recall
            rationale = f"score {score:.2f} above recall threshold"
        elif blindspot:
            # Symmetric verification: an unexplained blind-spot hit blocks a
            # clean negative (miss-averse asymmetry, axiom A5).
            decision = CaseDecision.defer_to_human
            rationale = (
                f"{len(blindspot)} unexplained region(s) from blind-spot re-search"
            )
        else:
            decision = CaseDecision.no_recall
            rationale = f"score {score:.2f} below deferral band"
        return Adjudication(
            per_candidate=per_candidate,
            decision=decision,
            rationale=rationale,
            cited_evidence=[c.get("evidence_ref", "") for c in request["candidates"]],
        )


class HarnessPipeline:
    def __init__(
        self,
        toolbelt: Toolbelt,
        adjudicator: Adjudicator | None = None,
        head: LogisticHead | None = None,
        conformal: MondrianConformal | None = None,
        consistency_reads: int = 5,
        min_reproduced: int = 3,
        zoom_margin: float = 0.75,
        zoom_score_ratio: float = 0.4,
        max_zoom_crops: int = 8,
        fp_max_aspect_ratio: float = 3.0,
        blindspot_min_score: float = 0.55,
        policy_id: str = "",
        router: DifficultyRouter | None = None,
        escalation_adjudicator: Adjudicator | None = None,
    ) -> None:
        if escalation_adjudicator is not None and router is None:
            raise ValueError(
                "escalation_adjudicator requires a router (U2): without lanes "
                "there is no deferral band for the LLM to rescue"
            )
        self.tools = toolbelt
        self.adjudicator = adjudicator or RuleBasedAdjudicator()
        # U2 cascade, opt-in: ``adjudicator`` answers the routine lane (the
        # rule-based default), ``escalation_adjudicator`` (typically the
        # LLM) answers only cases the router escalates. ``router is None``
        # reproduces the pre-U2 pipeline exactly — same calls, same costs.
        self.router = router
        self.escalation_adjudicator = escalation_adjudicator
        self.head = head
        self.conformal = conformal
        self.consistency_reads = consistency_reads
        self.min_reproduced = min_reproduced
        self.zoom_margin = zoom_margin
        self.zoom_score_ratio = zoom_score_ratio
        self.max_zoom_crops = max_zoom_crops
        self.fp_max_aspect_ratio = fp_max_aspect_ratio
        self.blindspot_min_score = blindspot_min_score
        # Which registered policy this pipeline embodies (U4/S2): stamped on
        # every CaseReport so A/B arms, bank replays, and incident forensics
        # stay attributable. Empty for ad-hoc, unregistered pipelines.
        self.policy_id = policy_id

    # -- stages ----------------------------------------------------------
    def preflight_qc(self, pixels: np.ndarray) -> QCVerdict:
        """Technical adequacy before any diagnostic reasoning (2G)."""
        dynamic_range = float(pixels.max() - pixels.min())
        saturated = float((pixels >= 0.999).mean())
        if dynamic_range < 0.05:
            return QCVerdict.inadequate_defer
        if saturated > 0.5:
            return QCVerdict.degraded
        return QCVerdict.adequate

    def _detect_with_shift(self, pixels: np.ndarray, dy: int, dx: int) -> list[dict]:
        """One TTA read: translate, detect, invert the transform on the boxes."""
        shifted = np.roll(np.roll(pixels, dy, axis=0), dx, axis=1)
        info = self.tools.store.put(shifted, kind="image", meta={"tta": [dy, dx]})
        result = self.tools.call("run_detector", image_handle=info.handle)
        out = []
        for cand in result["candidates"]:
            x0, y0, x1, y1 = cand["box"]
            out.append(
                {
                    "box": [x0 - dx, y0 - dy, x1 - dx, y1 - dy],
                    "score": cand["score"],
                    "evidence_ref": result["evidence_ref"],
                }
            )
        return out

    @staticmethod
    def _iou(a: list[int], b: list[int]) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def verify(self, pixels: np.ndarray, base_candidates: list[dict]) -> dict:
        """Self-consistency across TTA reads; IoU-cluster and count reproduction."""
        shifts = [(0, 0), (3, 0), (0, 3), (-3, 0), (0, -3)][: self.consistency_reads]
        reads = [self._detect_with_shift(pixels, dy, dx) for dy, dx in shifts]
        verified = []
        for i, cand in enumerate(base_candidates):
            reproduced = sum(
                1
                for read in reads
                if any(self._iou(cand["box"], other["box"]) >= 0.3 for other in read)
            )
            verified.append(
                {
                    "candidate_id": cand.get("candidate_id", f"c{i}"),
                    "box": cand["box"],
                    "score": cand["score"],
                    "evidence_ref": cand.get("evidence_ref", ""),
                    "reproduced": reproduced,
                    "reproduced_fraction": reproduced / len(reads),
                    "kept": reproduced >= self.min_reproduced,
                }
            )
        per_read_positive = [
            any(c["score"] >= 0.5 for c in read) for read in reads
        ]
        disagreement = float(np.mean(per_read_positive) * (1 - np.mean(per_read_positive)) * 4)
        return {"verified": verified, "disagreement_rate": round(disagreement, 4)}

    def zoom_verify(self, image_handle: str, verified: list[dict]) -> None:
        """Crop-and-re-detect every kept candidate at native resolution (A1).

        The clean-room look: a finding asserted on the whole image must also
        be found when only its own neighborhood is examined. Mutates each
        entry with ``zoom_confirmed`` and appends the zoom evidence ref.
        """
        crops_spent = 0
        for cand in verified:
            if not cand["kept"] or crops_spent >= self.max_zoom_crops:
                cand.setdefault("zoom_confirmed", cand["kept"])
                continue
            x0, y0, x1, y1 = cand["box"]
            mx = int((x1 - x0) * self.zoom_margin)
            my = int((y1 - y0) * self.zoom_margin)
            crop_res = self.tools.call(
                "crop_region",
                image_handle=image_handle,
                box=[x0 - mx, y0 - my, x1 + mx, y1 + my],
            )
            crops_spent += 1
            redetect = self.tools.call(
                "run_detector", image_handle=crop_res["crop_handle"]
            )
            ch, cw = crop_res["shape"]
            confirmed = False
            for zc in redetect["candidates"]:
                zx0, zy0, zx1, zy1 = zc["box"]
                cx, cy = (zx0 + zx1) / 2, (zy0 + zy1) / 2
                near_center = abs(cx - cw / 2) < cw / 3 and abs(cy - ch / 2) < ch / 3
                if near_center and zc["score"] >= self.zoom_score_ratio * cand["score"]:
                    confirmed = True
                    break
            cand["zoom_confirmed"] = confirmed
            cand["zoom_evidence_ref"] = redetect["evidence_ref"]
            if not confirmed:
                cand["kept"] = False

    def fp_hunt(self, image_handle: str, verified: list[dict], pixel_spacing_mm) -> None:
        """FP-hunter (A5): veto only with a NAMED alternative, never mere doubt."""
        for cand in verified:
            if not cand["kept"]:
                continue
            seg = self.tools.call(
                "segment",
                image_handle=image_handle,
                box=cand["box"],
                pixel_spacing_mm=list(pixel_spacing_mm),
            )
            veto = None
            if not seg.get("found"):
                veto = f"no coherent component at proposed location ({seg.get('reason')})"
            elif seg["aspect_ratio"] > self.fp_max_aspect_ratio:
                veto = (
                    f"elongated background ridge, not a mass "
                    f"(aspect ratio {seg['aspect_ratio']} > {self.fp_max_aspect_ratio})"
                )
            if veto:
                cand["kept"] = False
                cand["veto_reason"] = veto
                cand["veto_evidence_ref"] = seg["evidence_ref"]
            else:
                cand["segment_evidence_ref"] = seg.get("evidence_ref", "")
                cand["equivalent_diameter_mm"] = seg.get("equivalent_diameter_mm")

    def fn_hunt(self, image_handle: str, examined_boxes: list[list[int]]) -> list[dict]:
        """FN-hunter (A5): second detector family re-searches unexamined regions."""
        result = self.tools.call(
            "run_detector", image_handle=image_handle, profile="blindspot"
        )
        blindspot = []
        for cand in result["candidates"]:
            if cand["score"] < self.blindspot_min_score:
                continue
            bx0, by0, bx1, by1 = cand["box"]
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            covered = any(
                (x0 <= cx <= x1 and y0 <= cy <= y1)
                or self._iou(cand["box"], [x0, y0, x1, y1]) > 0.1
                for x0, y0, x1, y1 in examined_boxes
            )
            if not covered:
                blindspot.append(
                    {"box": cand["box"], "score": cand["score"],
                     "evidence_ref": result["evidence_ref"]}
                )
        return blindspot

    def aggregate(self, pixels: np.ndarray, verified: list[dict]) -> float:
        """Case-level suspicion score in [0,1]; calibrated head when available."""
        kept = [c for c in verified if c["kept"]]
        detector_score = max((c["score"] for c in kept), default=0.0)
        if self.head is not None:
            emb = embed_crop(pixels).reshape(1, -1)
            head_score = float(self.head.predict_proba(emb)[0])
            return round(0.5 * detector_score + 0.5 * head_score, 4)
        return round(float(detector_score), 4)

    # -- the whole case --------------------------------------------------
    def run_case(self, case_id: str, pixels: np.ndarray, pixel_spacing_mm=(0.1, 0.1)) -> CaseReport:
        meter = getattr(self.tools, "meter", None)
        if meter is not None:
            meter.start_case(case_id)
        info = self.tools.store.put(pixels, kind="image", meta={"case_id": case_id})
        qc = self.preflight_qc(pixels)
        if qc == QCVerdict.inadequate_defer:
            self.tools.call(
                "submit_review", case_id=case_id, reason="failed preflight QC", ranked_regions=[]
            )
            return CaseReport(
                case_id=case_id, policy_id=self.policy_id, qc=qc, findings=[],
                decision=CaseDecision.defer_to_human,
                score=0.5, disagreement_rate=1.0, deferral_reason="preflight QC",
                cost=meter.finish_case().as_dict() if meter is not None else None,
            )

        detect = self.tools.call("run_detector", image_handle=info.handle)
        for i, cand in enumerate(detect["candidates"]):
            cand["evidence_ref"] = detect["evidence_ref"]
        examined = [c["box"] for c in detect["candidates"]]

        # U2 routing: probe the blindspot early (the same call the unrouted
        # pipeline makes later, on the same inputs) and decide the lane as a
        # pure function of cheap features. Lane and blindspot never depend
        # on anything the LLM said.
        route: RouteRecord | None = None
        blindspot: list[dict] | None = None
        if self.router is not None:
            blindspot = self.fn_hunt(info.handle, examined)
            lane = self.router.route_after_detect(detect["candidates"], blindspot)
            route = RouteRecord(case_id=case_id, lane=lane.value)

        if route is not None and route.lane == Lane.fast_negative.value:
            # Fast-negative lane (E3): skip TTA/zoom/FP-hunt entirely. The
            # sub-threshold proposals ride along unkept so the adjudicator
            # and report still see them; ``aggregate`` scores zero kept
            # candidates (plus the calibrated head), and the conformal
            # safety net below still applies — the rule-out arm is cheap,
            # never unguarded.
            consistency = {
                "verified": [
                    {
                        "candidate_id": cand.get("candidate_id", f"c{i}"),
                        "box": cand["box"],
                        "score": cand["score"],
                        "evidence_ref": cand.get("evidence_ref", ""),
                        "reproduced": 0,
                        "reproduced_fraction": 0.0,
                        "kept": False,
                    }
                    for i, cand in enumerate(detect["candidates"])
                ],
                "disagreement_rate": 0.0,
            }
        else:
            # Early exit (E5, routed only): zero proposals make TTA vacuous,
            # and zoom/FP-hunt act only on kept candidates — skip stages
            # that can no longer change the decision.
            if self.router is not None and not detect["candidates"]:
                consistency = {"verified": [], "disagreement_rate": 0.0}
            else:
                consistency = self.verify(pixels, detect["candidates"])

            # Inference stack v1: clean-room zoom, then symmetric verification.
            if self.router is None or any(c["kept"] for c in consistency["verified"]):
                self.zoom_verify(info.handle, consistency["verified"])
                self.fp_hunt(info.handle, consistency["verified"], pixel_spacing_mm)
            if blindspot is None:
                blindspot = self.fn_hunt(info.handle, examined)

        case_score = self.aggregate(pixels, consistency["verified"])

        request = {
            "case_id": case_id,
            "qc": qc.value,
            "candidates": consistency["verified"],
            "consistency": {
                "case_score": case_score,
                "disagreement_rate": consistency["disagreement_rate"],
            },
            "blindspot_candidates": blindspot,
            "atlas_neighbors": [],
            "guideline_notes": [],
        }
        if route is not None:
            request["lane"] = route.lane
        adjudication = self.adjudicator.adjudicate(request)
        conformal_ambiguous = self.conformal is not None and self.conformal.is_ambiguous(
            case_score
        )

        # U2/U7 escalation cascade: rules -> distilled student -> LLM. The
        # escalation decision is a pure function; the LLM is reached only
        # through the router's circuit breaker, and a case denied by the cap
        # keeps its rule verdict (never a truncated half-answer).
        if route is not None:
            route.escalation_wanted = self.router.escalate_to_llm(
                adjudication, conformal_ambiguous, blindspot
            )
            if route.escalation_wanted:
                student = self.router.student_verdict(request)
                if student is not None:
                    decision, confidence = student
                    route.student_absorbed = True
                    adjudication = Adjudication(
                        per_candidate=adjudication.per_candidate,
                        decision=decision,
                        rationale=(
                            f"distilled student verdict at confidence {confidence:.3f} "
                            "(at or above the promoted threshold)"
                        ),
                        cited_evidence=adjudication.cited_evidence,
                    )
                elif self.escalation_adjudicator is not None:
                    if self.router.admit_llm():
                        route.lane = Lane.hard.value
                        route.llm_invoked = True
                        request["lane"] = route.lane
                        adjudication = self.escalation_adjudicator.adjudicate(request)
                    else:
                        route.llm_denied_by_cap = True
            # Routing telemetry: one typed ledger entry per routed case, so
            # lane mix and escalation outcomes are auditable per run (S8:
            # written by code, never a registered tool).
            self.tools.ledger.append("route", route.as_dict())

        # Conformal policy layer (A13): an ambiguous or empty prediction set
        # overrides any confident-sounding decision with deferral — over
        # every lane, the LLM's and the student's verdicts included.
        if (
            self.conformal is not None
            and adjudication.decision != CaseDecision.defer_to_human
            and conformal_ambiguous
        ):
            pset = sorted(self.conformal.predict_set(case_score))
            adjudication = Adjudication(
                per_candidate=adjudication.per_candidate,
                decision=CaseDecision.defer_to_human,
                rationale=(
                    f"conformal prediction set {pset} is not a singleton "
                    f"at score {case_score:.2f}"
                ),
                cited_evidence=adjudication.cited_evidence,
            )
        self.tools.ledger.append("decision", adjudication.model_dump(mode="json"))

        findings = []
        for cand in consistency["verified"]:
            if not cand["kept"]:
                continue
            measure = self.tools.call(
                "measure",
                image_handle=info.handle,
                box=cand["box"],
                pixel_spacing_mm=list(pixel_spacing_mm),
            )
            refs = [cand["evidence_ref"], measure["evidence_ref"]]
            for extra in ("zoom_evidence_ref", "segment_evidence_ref"):
                if cand.get(extra):
                    refs.append(cand[extra])
            findings.append(
                Finding(
                    finding_id=cand["candidate_id"],
                    box=tuple(int(v) for v in cand["box"]),
                    size_mm=measure["long_axis_mm"],
                    detector_score=cand["score"],
                    assessment=adjudication.per_candidate.get(
                        cand["candidate_id"], Assessment.uncertain
                    ),
                    evidence_refs=refs,
                )
            )

        if adjudication.decision == CaseDecision.defer_to_human:
            self.tools.call(
                "submit_review",
                case_id=case_id,
                reason=adjudication.rationale,
                ranked_regions=[
                    {"box": f.box, "size_mm": f.size_mm, "score": f.detector_score}
                    for f in findings
                ],
            )

        report = CaseReport(
            case_id=case_id,
            policy_id=self.policy_id,
            qc=qc,
            lane=route.lane if route is not None else "",
            findings=findings,
            decision=adjudication.decision,
            score=case_score,
            disagreement_rate=consistency["disagreement_rate"],
            deferral_reason=(
                adjudication.rationale
                if adjudication.decision == CaseDecision.defer_to_human
                else ""
            ),
            evidence_refs=[detect["evidence_ref"]],
            cost=meter.finish_case().as_dict() if meter is not None else None,
        )
        self.tools.ledger.append("claim", report.model_dump(mode="json"))
        return report
