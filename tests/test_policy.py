"""Policy registry (U4): content-hashed configs, lineage, champion pointer, rollback."""

from pathlib import Path

import pytest

from oncoharness.ledger import EvidenceLedger
from oncoharness.phantom import generate_dataset
from oncoharness.policy import (
    PolicyConfig,
    PolicyRegistry,
    default_policy,
    pipeline_from_policy,
)
from oncoharness.store import ArtifactStore
from oncoharness.tools import Toolbelt

PASSING_SUMMARY = "[PASS] primary_metric_non_inferiority: ...\nGATE: PASS"
FAILING_SUMMARY = "[FAIL] calibration_ece: ...\nGATE: FAIL"


def _registry(tmp_path) -> PolicyRegistry:
    return PolicyRegistry(tmp_path / "registry")


def test_register_get_roundtrip_and_content_hash_identity(tmp_path):
    reg = _registry(tmp_path)
    cfg = default_policy()
    pid = reg.register(cfg, parent_id=None, note="generation zero")
    assert reg.get(pid) == cfg
    # Identical config -> identical id (and idempotent registration).
    assert PolicyConfig(**cfg.model_dump()).policy_id() == pid
    assert reg.register(cfg) == pid
    # One parameter changed -> a different policy, a different name.
    tweaked = cfg.model_dump()
    tweaked["pipeline"]["consistency_reads"] = 7
    new_pid = reg.register(PolicyConfig(**tweaked), parent_id=pid, note="more reads")
    assert new_pid != pid
    assert reg.get(new_pid).pipeline["consistency_reads"] == 7


def test_register_refuses_unknown_parent(tmp_path):
    reg = _registry(tmp_path)
    with pytest.raises(KeyError):
        reg.register(default_policy(), parent_id="deadbeef0000")


def test_promote_refuses_without_passing_gate(tmp_path):
    reg = _registry(tmp_path)
    pid = reg.register(default_policy())
    with pytest.raises(PermissionError):
        reg.promote(pid, gate_summary=FAILING_SUMMARY, gate_passed=False)
    # A caller lying about the flag is caught by the summary text itself.
    with pytest.raises(PermissionError):
        reg.promote(pid, gate_summary=FAILING_SUMMARY, gate_passed=True)
    with pytest.raises(PermissionError):
        reg.promote(pid, gate_summary="", gate_passed=True)
    assert reg.champion() is None


def test_promote_then_rollback_restores_pointer_and_records_lineage(tmp_path):
    reg = _registry(tmp_path)
    gen0 = reg.register(default_policy(), note="gen0")
    tweaked = default_policy().model_dump()
    tweaked["adjudicator"]["recall_threshold"] = 0.7
    gen1 = reg.register(PolicyConfig(**tweaked), parent_id=gen0, note="gen1")

    reg.promote(gen0, PASSING_SUMMARY, gate_passed=True)
    reg.promote(gen1, PASSING_SUMMARY, gate_passed=True)
    assert reg.champion() == gen1

    reg.rollback(gen0, reason="gen1 regressed on-call; drill T-5.6")
    assert reg.champion() == gen0

    events = [(e["kind"], e["payload"]["policy_id"]) for e in reg.lineage()]
    assert ("promote", gen1) in events and ("rollback", gen0) in events
    # The rollback event names the champion it displaced.
    rb = [e for e in reg.lineage() if e["kind"] == "rollback"][-1]
    assert rb["payload"]["previous_champion"] == gen1


def test_rollback_refuses_never_champion_target_and_empty_reason(tmp_path):
    reg = _registry(tmp_path)
    gen0 = reg.register(default_policy())
    with pytest.raises(PermissionError):
        reg.rollback(gen0, reason="never promoted: this would dodge the gate")
    reg.promote(gen0, PASSING_SUMMARY, gate_passed=True)
    with pytest.raises(ValueError):
        reg.rollback(gen0, reason="   ")


def test_lineage_chain_verifies_and_detects_tampering(tmp_path):
    reg = _registry(tmp_path)
    pid = reg.register(default_policy())
    reg.promote(pid, PASSING_SUMMARY, gate_passed=True)
    assert reg.verify_chain()
    # Rewrite one byte of history: the chain must break from that point on.
    path = reg.lineage_ledger.path
    lines = path.read_text().splitlines()
    lines[0] = lines[0].replace("register", "promoted")
    path.write_text("\n".join(lines) + "\n")
    assert not EvidenceLedger(path).verify_chain()


def test_policy_file_tampering_is_detected_on_get(tmp_path):
    reg = _registry(tmp_path)
    pid = reg.register(default_policy())
    path = reg.policies_dir / f"{pid}.json"
    path.write_text(path.read_text().replace("0.65", "0.99"))
    with pytest.raises(RuntimeError):
        reg.get(pid)


def test_prompt_registration_is_content_addressed(tmp_path):
    reg = _registry(tmp_path)
    sha = reg.register_prompt("Adjudicate strictly from cited evidence.")
    assert reg.register_prompt("Adjudicate strictly from cited evidence.") == sha
    assert reg.get_prompt(sha) == "Adjudicate strictly from cited evidence."
    cfg = default_policy(prompt_ids={"adjudicator": sha})
    assert cfg.policy_id() != default_policy().policy_id()


def test_run_case_stamps_policy_id(tmp_path):
    cfg = default_policy().model_dump()
    cfg["pipeline"].update({"consistency_reads": 3, "min_reproduced": 2})
    cfg = PolicyConfig(**cfg)
    reg = _registry(tmp_path)
    pid = reg.register(cfg)

    tb = Toolbelt(ArtifactStore(tmp_path / "artifacts"), EvidenceLedger(tmp_path / "ledger.jsonl"))
    pipeline = pipeline_from_policy(cfg, tb)
    case = generate_dataset(n_patients=2, images_per_patient=1, prevalence=0.5, seed=3)[0]
    report = pipeline.run_case(case.case_id, case.pixels)
    assert report.policy_id == pid
    # Backwards compatibility: an unregistered ad-hoc pipeline stamps nothing.
    from oncoharness.state_machine import HarnessPipeline

    bare = HarnessPipeline(tb, consistency_reads=3, min_reproduced=2)
    assert bare.run_case(case.case_id, case.pixels).policy_id == ""


def test_pipeline_from_policy_bounds_the_reachable_family(tmp_path):
    tb = Toolbelt(ArtifactStore(tmp_path / "artifacts"), EvidenceLedger(tmp_path / "ledger.jsonl"))
    rogue = PolicyConfig(pipeline={"exec_hook": "os.system"}, adjudicator={"kind": "rule"})
    with pytest.raises(ValueError, match="reachable family"):
        pipeline_from_policy(rogue, tb)
    rogue_adj = PolicyConfig(adjudicator={"kind": "rule", "shell": "/bin/sh"})
    with pytest.raises(ValueError, match="reachable family"):
        pipeline_from_policy(rogue_adj, tb)
    with pytest.raises(ValueError, match="kind"):
        pipeline_from_policy(PolicyConfig(adjudicator={"kind": "freeform_code"}), tb)


def test_only_policy_module_writes_lineage():
    """v1 stand-in for the service-account boundary (T-3.3 discipline, S8).

    No harness/LLM code path may touch the lineage ledger except through
    PolicyRegistry: the string 'lineage.jsonl' must appear nowhere in the
    package outside policy.py, and the Toolbelt registry must expose no
    tool that could reach it.
    """
    src_root = Path(__file__).resolve().parents[1] / "src" / "oncoharness"
    offenders = [
        p.name
        for p in src_root.rglob("*.py")
        if p.name != "policy.py" and "lineage.jsonl" in p.read_text()
    ]
    assert offenders == []
    import oncoharness.tools as tools_mod

    assert "PolicyRegistry" not in Path(tools_mod.__file__).read_text()
