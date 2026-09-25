"""The compiler's decisions: what becomes a step, a detector, a typed input, a checkpoint, a risk level."""

from __future__ import annotations

from pathlib import Path

from mm.agent.loop import DiscoveryResult, RecordedStep
from mm.artifact.compiler import compile_run
from mm.surface.base import ActionType, RoleStrategy, Target

MAIN = {"": "http://app/core/main.jsp", "main": "http://app/core/form.jsp"}


def _target(role: str, name: str) -> Target:
    return Target(frame_path=["main"], strategies=[RoleStrategy(role=role, name=name)],
                  description=f'{role} "{name}" in frame main')


def _step(action: ActionType, role: str, name: str, value: str | None = None, *, after: dict[str, str] | None = None,
          interruption: bool = False, options: list[str] | None = None, output: str | None = None) -> RecordedStep:
    return RecordedStep(action, f"{action} {name}", _target(role, name), value, output, MAIN, after or MAIN,
                        interruption, options)


def _run() -> DiscoveryResult:
    return DiscoveryResult(
        run_id="discover-test", status="success", summary="ok", goal="open a Holiday Club for 10234 with 25.00",
        entry_url="http://app/login", inputs={"member_id": "10234", "account_type": "Holiday Club", "deposit": "25.00"},
        outputs={"confirmation_number": "CNF123456"}, provider="groq", model="m",
        steps=[
            _step(ActionType.FILL, "textbox", "Member #", "{{member_id}}"),
            _step(ActionType.CLICK, "button", "Acknowledge", interruption=True),  # a popup the agent dismissed
            _step(ActionType.SELECT, "combobox", "Account Type", "{{account_type}}",
                  options=["-- select --", "Share Savings", "Holiday Club"]),
            _step(ActionType.FILL, "textbox", "Opening Deposit", "{{deposit}}"),
            _step(ActionType.CLICK, "button", "Confirm", after={**MAIN, "main": "http://app/core/done.jsp?c=1"}),
            _step(ActionType.EXTRACT, "cell", "Confirmation", output="confirmation_number"),
        ],
    )


def test_dismissed_popup_becomes_a_recoverable_detector_not_a_step(tmp_path: Path) -> None:
    cap = compile_run(_run(), "app.member.open_account", packs_dir=tmp_path)
    assert all("acknowledge" not in s.id for s in cap.steps)
    learned = [d for d in cap.detectors if d.id.startswith("learned_")]
    assert len(learned) == 1 and learned[0].class_ == "recoverable"
    assert learned[0].source == "discovery:discover-test"


def test_input_types_are_inferred_from_values_and_dropdown_options(tmp_path: Path) -> None:
    cap = compile_run(_run(), "app.member.open_account", packs_dir=tmp_path)
    assert cap.inputs["member_id"].pattern == r"^[0-9]+$"
    assert cap.inputs["account_type"].enum == ["Share Savings", "Holiday Club"]
    assert cap.inputs["deposit"].type == "decimal"


def test_checkpoints_and_risk(tmp_path: Path) -> None:
    cap = compile_run(_run(), "app.member.open_account", packs_dir=tmp_path)
    by_name = {s.id.split("_", 1)[1]: s for s in cap.steps}
    # no navigation: wait for the next step's control to be there
    assert [c.kind for c in by_name["fill_member"].expect] == ["target_present"]
    # navigation: the frame must land on the recorded page, whatever the query string
    confirm = by_name["click_confirm"]
    assert confirm.risk == "irreversible"
    assert confirm.expect[0].kind == "url_matches" and confirm.expect[0].pattern == r"/core/done\.jsp(\?|$)"
    assert by_name["select_account_type"].risk == "safe"


def test_pack_detectors_are_copied_in_with_their_source(tmp_path: Path) -> None:
    (tmp_path / "app.yaml").write_text(
        "pack: app\nversion: 2.1.0\ndetectors:\n"
        "  - {id: not_found, class: business_outcome, code: NOT_FOUND,"
        " when: [{kind: text_visible, any_frame: true, pattern: 'No match'}]}\n")
    cap = compile_run(_run(), "app.member.open_account", packs_dir=tmp_path)
    assert cap.outcomes == ["NOT_FOUND"]
    assert cap.detectors[0].source == "pack:app@2.1.0"


def test_no_concrete_input_value_is_stored(tmp_path: Path) -> None:
    cap = compile_run(_run(), "app.member.open_account", packs_dir=tmp_path)
    dumped = cap.model_dump_json()
    assert "10234" not in dumped and "25.00" not in dumped
    assert "{{member_id}}" in cap.summary
