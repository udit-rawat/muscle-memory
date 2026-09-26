"""Regenerate /evidence/ from the committed capabilities, reproducibly.

    uv run python scripts/make_evidence.py

Starts its own mock banks (tenant A and tenant B) on spare ports, runs every replay scenario through the
real CLI (`mm replay`, production defaults: bound policy, no traces, masked screenshots), and copies each
run's folder into evidence/NN_name/ with the exact command and its result. The two discovery runs are the
ones that produced the committed capabilities (found through `provenance.discovery_run_id`); they are
copied, not re-run, because re-running an LLM would produce a different run than the one the artifact
came from.

Nothing here needs an LLM key. Values a caller would receive (balances, confirmation numbers) are masked
in the copied result.json: evidence shows the shape of every answer, never the regulated value.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from mm.artifact import store  # noqa: E402
from mm.artifact.approval import approval_path  # noqa: E402
from mm.redact import mask_value  # noqa: E402

EVIDENCE = ROOT / "evidence"
BALANCE = store.latest_path("corebank.member.get_savings_balance", ROOT / "capabilities")
OPEN = store.latest_path("corebank.member.open_sub_account", ROOT / "capabilities")
OPEN_INPUTS = ["-p", "member_id=10871", "-p", "account_type=Holiday Club", "-p", "deposit=25", "-p", "nickname=Fund"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Bank:
    def __init__(self, tenant: str) -> None:
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc = subprocess.Popen([sys.executable, "-m", "mm.cli", "mockbank", "--tenant", tenant,
                                      "--port", str(self.port)], cwd=ROOT, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                httpx.get(f"{self.url}/login", timeout=1)
                return
            except httpx.HTTPError:
                time.sleep(0.1)
        raise RuntimeError(f"mock bank ({tenant}) did not start")

    def faults(self, *names: str) -> None:
        httpx.post(f"{self.url}/__control/reset")
        if names:
            httpx.post(f"{self.url}/__control/faults", json={"faults": list(names)})

    def stop(self) -> None:
        self.proc.terminate()


def replay_scenario(name: str, what: str, artifact: Path, args: list[str], bank: Bank, runs: Path,
                    faults: tuple[str, ...] = (), extra_env: dict[str, str] | None = None,
                    note: str = "") -> dict[str, Any]:
    bank.faults(*faults)
    cmd = ["mm", "replay", _rel(artifact), *args, "--base-url", bank.url, "--headless"]
    env = {**os.environ, "MM_RUNS_DIR": str(runs), **(extra_env or {})}
    before = set(runs.glob("*"))
    out = subprocess.run([sys.executable, "-m", "mm.cli", *cmd[1:]], capture_output=True, text=True, env=env,
                         cwd=ROOT)
    result = json.loads(out.stdout)
    run_dir = next(iter(set(runs.glob("*")) - before))
    dest = EVIDENCE / name
    shutil.copytree(run_dir, dest)
    (dest / "result.json").write_text(json.dumps(_masked(result), indent=2) + "\n")
    header = f"# mock bank faults: {', '.join(faults)}\n" if faults else ""
    header += f"# {note}\n" if note else ""
    (dest / "command.txt").write_text(header + " ".join(_quote(c) for c in cmd) + "\n")
    print(f"  {name:44} {result['status']:17} {result.get('kind') or result.get('code') or ''}")
    return {"name": name, "what": what, "status": result["status"],
            "detail": result.get("kind") or result.get("code") or ", ".join(result.get("outputs", {}))}


def discovery_scenario(name: str, what: str, artifact: Path) -> dict[str, Any]:
    cap = store.load(artifact)
    assert cap.provenance is not None
    src = ROOT / "runs" / cap.provenance.discovery_run_id
    if not src.exists():
        raise SystemExit(f"the discovery run {src} behind {_rel(artifact)} is not on this machine; "
                         "keep the existing evidence folder for it")
    dest = EVIDENCE / name
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("trace.zip"))
    status = next(json.loads(line) for line in (dest / "events.jsonl").read_text().splitlines()
                  if '"discovery_end"' in line)["status"]
    print(f"  {name:44} {status}")
    return {"name": name, "what": what, "status": status, "detail": f"{cap.id} {cap.version}"}


def stability(name: str, bank: Bank, runs: Path, n: int = 20) -> dict[str, Any]:
    bank.faults()
    results = []
    for _ in range(n):
        out = subprocess.run([sys.executable, "-m", "mm.cli", "replay", str(BALANCE), "-p", "member_id=10871",
                              "--base-url", bank.url, "--headless"], capture_output=True, text=True, cwd=ROOT,
                             env={**os.environ, "MM_RUNS_DIR": str(runs)})
        r = json.loads(out.stdout)
        results.append({"status": r["status"], "drift": r.get("drift", []),
                        "ms": sum(s["duration_ms"] for s in r["steps"])})
    ok = sum(r["status"] == "success" for r in results)
    drifted = sum(bool(r["drift"]) for r in results)
    summary = {"capability": f"{store.load(BALANCE).id} {store.load(BALANCE).version}", "runs": n, "success": ok,
               "with_drift": drifted, "step_ms_min": min(r["ms"] for r in results),
               "step_ms_max": max(r["ms"] for r in results)}
    dest = EVIDENCE / name
    dest.mkdir(parents=True)
    (dest / "stability.json").write_text(json.dumps({"summary": summary, "runs": results}, indent=2) + "\n")
    print(f"  {name:44} {ok}/{n} success, {drifted} with drift")
    return {"name": name, "what": f"{n} consecutive replays", "status": f"{ok}/{n} success",
            "detail": f"{drifted} with drift"}


def main() -> None:
    if not approval_path(OPEN).exists():
        raise SystemExit(f"{_rel(OPEN)} has no approval yet: run `uv run mm approve {_rel(OPEN)} --by <you>` first")
    if EVIDENCE.exists():
        shutil.rmtree(EVIDENCE)
    EVIDENCE.mkdir()
    runs = Path(tempfile.mkdtemp(prefix="mm-evidence-"))
    unapproved = runs / "unapproved" / OPEN.name  # the same capability, without its approval next to it
    unapproved.parent.mkdir()
    shutil.copy(OPEN, unapproved)
    a, b = Bank("tenant_a"), Bank("tenant_b")
    handoff_env = {"MM_OPERATOR_PORT": str(free_port())}
    rows: list[dict[str, Any]] = []
    try:
        print("discovery (copied from the runs that produced the committed capabilities):")
        rows.append(discovery_scenario("01_discovery_savings_balance",
                                       "LLM discovery with an unknown popup on: the popup becomes a learned "
                                       "recoverable detector, not a step", BALANCE))
        rows.append(discovery_scenario("02_discovery_open_sub_account",
                                       "LLM discovery that pauses for a human approval before Confirm", OPEN))
        print("replay (no LLM):")
        bal = ["-p", "member_id=10871"]
        rows.append(replay_scenario("03_replay_success", "clean deterministic replay", BALANCE, bal, a, runs))
        rows.append(replay_scenario("04_replay_member_not_found", "a legitimate answer, not a crash", BALANCE,
                                    ["-p", "member_id=99999"], a, runs))
        rows.append(replay_scenario("05_replay_recovered_notice", "known interstitial dismissed by the pack",
                                    BALANCE, bal, a, runs, faults=("notice",)))
        rows.append(replay_scenario("06_replay_recovered_session_expiry", "session expires mid-flow; safe restart",
                                    BALANCE, bal, a, runs, faults=("session_expired_on_detail",)))
        rows.append(replay_scenario("07_replay_learned_popup", "popup recovered by the detector learned in 01",
                                    BALANCE, bal, a, runs, faults=("survey",)))
        rows.append(replay_scenario("08_replay_app_error", "hard failure with step, expected, observed",
                                    BALANCE, bal, a, runs, faults=("error500",)))
        rows.append(replay_scenario("09_replay_no_savings_account", "value absent: never read from another row",
                                    BALANCE, ["-p", "member_id=13100"], a, runs))
        rows.append(replay_scenario("10_replay_input_invalid", "rejected before any browser opens", BALANCE,
                                    ["-p", "member_id=12ab"], a, runs))
        rows.append(replay_scenario("11_replay_unapproved_irreversible", "Confirm refused without an approval",
                                    unapproved, OPEN_INPUTS, a, runs,
                                    note=f"identical copy of {_rel(OPEN)}, without its approval file"))
        rows.append(replay_scenario("12_replay_approved_commit", "approved capability commits", OPEN,
                                    OPEN_INPUTS, a, runs))
        rows.append(replay_scenario("13_replay_handoff_unknown_popup",
                                    "unknown popup -> operator takes the live session, fixes it, hands back",
                                    OPEN, [*OPEN_INPUTS, "--simulate-operator", "click:Maybe Later"], a, runs,
                                    faults=("survey",), extra_env=handoff_env))
        rows.append(replay_scenario("14_replay_unsafe_to_repeat", "expiry right after a mutating submit: no resubmit",
                                    OPEN, OPEN_INPUTS, a, runs, faults=("session_expired_on_submit",)))
        rows.append(replay_scenario("15_tenant_b_without_profile", "tenant-A capability on tenant B: refuses the "
                                    "wrong link instead of clicking it", BALANCE, bal, b, runs))
        rows.append(replay_scenario("16_tenant_b_with_profile", "same capability + tenant profile: succeeds",
                                    BALANCE, [*bal, "--tenant", "tenant_b"], b, runs))
        print("stability:")
        rows.append(stability("17_stability_20_runs", a, runs))
    finally:
        a.stop()
        b.stop()
        shutil.rmtree(runs, ignore_errors=True)
    _index(rows)


def _index(rows: list[dict[str, Any]]) -> None:
    lines = ["# Evidence", "",
             "Generated by `uv run python scripts/make_evidence.py` from the committed capabilities "
             f"(`{_rel(BALANCE)}`, `{_rel(OPEN)}`). Each folder holds the run's `events.jsonl` (structured log), "
             "masked screenshots, and for replays `command.txt` and `result.json` (values masked); discovery "
             "folders also hold `prompts/` (exactly what the model received) and `artifact.yaml`.", "",
             "| # | Scenario | What it shows | Result |", "|---|---|---|---|"]
    for r in rows:
        num, _, title = r["name"].partition("_")
        lines.append(f"| {num} | [{title.replace('_', ' ')}]({r['name']}/) | {r['what']} | "
                     f"`{r['status']}` {r['detail']} |")
    (EVIDENCE / "README.md").write_text("\n".join(lines) + "\n")


def _masked(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("outputs"):
        result["outputs"] = {k: mask_value(str(v)) for k, v in result["outputs"].items()}
        result["_note"] = "outputs masked for evidence; the caller receives the actual values"
    return result


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _quote(arg: str) -> str:
    return f'"{arg}"' if " " in arg else arg


if __name__ == "__main__":
    main()
