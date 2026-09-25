from pathlib import Path

import pytest
from pydantic import ValidationError

from mm.artifact import store
from mm.artifact.schema import Capability

FIXTURE = store.latest_path("corebank.member.get_savings_balance",
                            Path(__file__).resolve().parent.parent / "capabilities")


def test_recorded_artifact_round_trips(tmp_path: Path) -> None:
    cap = store.load(FIXTURE)
    assert cap.inputs.keys() == {"member_id"} and cap.outputs.keys() == {"savings_balance"}
    assert store.load(store.save(cap, tmp_path)) == cap


def test_artifact_holds_no_concrete_values() -> None:
    text = FIXTURE.read_text()
    for leaked in ("10234", "2,450.17", "2450.17", "change-me-local-only", "operator1"):
        assert leaked not in text


def _broken(**changes: object) -> dict[str, object]:
    data = store.load(FIXTURE).model_dump(mode="json")
    data.update(changes)
    return data


def test_undeclared_input_is_rejected() -> None:
    with pytest.raises(ValidationError, match="undeclared inputs"):
        Capability.model_validate(_broken(inputs={}))


def test_undeclared_secret_is_rejected() -> None:
    with pytest.raises(ValidationError, match="undeclared secrets"):
        Capability.model_validate(_broken(secrets=[]))


def test_output_never_extracted_is_rejected() -> None:
    data = _broken()
    data["outputs"] = {**data["outputs"], "phantom": {"type": "string"}}  # type: ignore[dict-item]
    with pytest.raises(ValidationError, match="never extracted"):
        Capability.model_validate(data)


def test_id_and_version_format() -> None:
    with pytest.raises(ValidationError):
        Capability.model_validate(_broken(id="Not A Dotted Id"))
    with pytest.raises(ValidationError):
        Capability.model_validate(_broken(version="1.0"))


def _detector(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {"id": "d", "class": "business_outcome", "code": "NOPE",
                               "when": [{"kind": "text_visible", "pattern": "x"}]}
    return {**base, **overrides}


def test_detector_rules() -> None:
    with pytest.raises(ValidationError, match="UPPER_SNAKE code"):
        Capability.model_validate(_broken(detectors=[_detector(code=None)], outcomes=[]))
    with pytest.raises(ValidationError, match="only recoverable"):
        Capability.model_validate(_broken(detectors=[_detector(handle=[{"action": "click", "target": None}])],
                                          outcomes=["NOPE"]))
    with pytest.raises(ValidationError, match="must list exactly"):
        Capability.model_validate(_broken(detectors=[_detector()], outcomes=[]))
    with pytest.raises(ValidationError, match="unknown steps"):
        Capability.model_validate(_broken(detectors=[_detector(after_steps=["s99"])], outcomes=["NOPE"]))


def test_business_outcomes_are_declared_in_the_contract() -> None:
    cap = store.load(FIXTURE)
    assert "MEMBER_NOT_FOUND" in cap.outcomes and "PERMISSION_DENIED" in cap.outcomes
    assert all(d.source.startswith(("pack:corebank@", "discovery:")) for d in cap.detectors)
