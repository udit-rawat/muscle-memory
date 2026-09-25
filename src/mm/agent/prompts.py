"""Prompt construction. Each turn is stateless (system + one user message) so token use stays flat:
the model gets the goal, a compact history, and the current screen, never the whole transcript.
"""

from __future__ import annotations

from collections.abc import Mapping

from mm.surface.base import Observation

SYSTEM = """\
You operate a legacy back-office web application for a credit union, on behalf of a staff operator.
You see the screen as a list of elements, each with a ref like [e12], a role, a name and its frame,
plus the visible text of each frame. Choose exactly ONE next action per turn.

Actions:
- click(ref)                       click a link or button
- fill(ref, value)                 type into a textbox (replaces its content)
- select(ref, value)               choose a dropdown option by its label
- extract(ref, output_name)        read the text of one element (usually a table cell) into a named output
- navigate(value=url)              only if no link gets you there
- done(summary)                    the goal is complete and every value it asks for has been extracted
- fail(summary)                    the goal cannot be completed (e.g. record not found, not authorized)
- request_human(summary)           you are stuck or blocked and need an operator

Rules:
- For task inputs, type the placeholder, e.g. {{member_id}}, not the raw value.
- For credentials, type {{secret:NAME}} using the listed secret names. Never guess credentials.
- Extract a single cell holding the value itself (e.g. the balance cell), not a whole row or label.
- If an unexpected dialog or notice blocks the page, dismiss it first and set "interruption": true on that click.
- If your last action did not change the screen, do something different.
- Keep 'thought' to one short sentence.
Respond with JSON only."""


def user_message(
    goal: str,
    inputs: Mapping[str, str],
    secret_names: list[str],
    extracted: list[str],
    history: list[str],
    obs: Observation,
) -> str:
    lines = [f"GOAL: {goal}", ""]
    if inputs:
        lines.append("INPUTS (type as placeholders): " + ", ".join(f"{{{{{k}}}}}={v}" for k, v in inputs.items()))
    if secret_names:
        lines.append("SECRETS available: " + ", ".join(f"{{{{secret:{n}}}}}" for n in secret_names))
    if extracted:
        # Names only: the values stay out of the prompt, and showing masked values made the model re-read them.
        lines.append("ALREADY EXTRACTED (do not extract again): " + ", ".join(extracted))
    lines += ["", "HISTORY:" if history else "HISTORY: (none, this is the first step)"]
    lines += history[-8:]
    lines += ["", f"SCREEN: {obs.title} | {obs.url}"]
    for f in obs.frames:
        where = "/".join(f.frame_path) or "top"
        lines.append(f"--- frame {where} text ---\n{f.text}")
    lines.append("--- elements ---")
    for e in obs.elements:
        where = f" @{'/'.join(e.frame_path)}" if e.frame_path else ""
        val = f" value={e.value!r}" if e.value else ""
        opts = f" options=[{' | '.join(e.options)}]" if e.options else ""
        lines.append(f"[{e.ref}] {e.role} \"{e.name}\"{val}{opts}{where}")
    if obs.truncated:
        lines.append(f"({obs.truncated} more elements not shown: the screen is larger than the observation limit)")
    return "\n".join(lines)
