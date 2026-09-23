from pathlib import Path

import pytest
from pydantic import ValidationError

from mm.artifact import store
from mm.artifact.schema import Capability

FIXTURE = Path(__file__).parent / "fixtures" / "get_savings_balance.yaml"


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
