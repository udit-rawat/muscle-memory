import pytest
from pydantic import ValidationError

from mm.agent.decision import for_screen

Screen = for_screen({"e1", "e2"}, {"member_id"}, {"PW"})


def test_valid_click() -> None:
    assert Screen(thought="t", action="click", ref="e1").ref == "e1"


@pytest.mark.parametrize("payload, error", [
    ({"action": "click", "ref": "e99"}, "not on the current screen"),
    ({"action": "fill", "ref": "e1"}, "requires 'value'"),
    ({"action": "fill", "ref": "e1", "value": "{{account}}"}, "unknown input"),
    ({"action": "fill", "ref": "e1", "value": "{{secret:TOKEN}}"}, "unknown secret"),
    ({"action": "extract", "ref": "e2", "output_name": "Savings Balance"}, "snake_case"),
    ({"action": "done"}, "requires 'summary'"),
])
def test_invalid_decisions_are_rejected(payload: dict[str, str], error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        Screen(thought="t", **payload)
