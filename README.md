# muscle-memory

An LLM works out how to complete a task inside a UI that has no API; the successful run is compiled into a
typed, versioned **capability**; the capability then replays **deterministically, with no model in the loop**,
returning a structured result: `success`, a `business_outcome` (e.g. `MEMBER_NOT_FOUND`), or a debuggable
`failure`. When it cannot proceed safely it hands the **same live session** to a human and takes it back.

The target is a deliberately legacy mock core-banking app (framesets, table layouts, no test IDs) with
injectable runtime faults. Design and trade-offs: [REPORT.md](REPORT.md). Evidence: [evidence/](evidence/README.md).

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run playwright install chromium
cp .env.example .env        # then add your keys (only needed for discovery)
uv run mm doctor --ping     # optional: checks both LLM providers
```

`.env` keys: `GROQ_API_KEY` (primary, Groq free tier) and `GEMINI_API_KEY` (fallback, Google AI Studio free
tier; `MM_FALLBACK_MODEL` is a comma-separated list tried in order). The mock bank credentials in
`.env.example` are local dummies.

### Without live services

Only **discovery** calls an LLM. Replay, the mock bank, the operator console and the whole test suite run
offline, with no API key:

```bash
uv run pytest -q                              # unit, browser-free control flow, live replays
uv run python scripts/make_evidence.py        # regenerates evidence/ from the committed capabilities
```

## Demo path

Terminal 1, the target application:

```bash
uv run mm mockbank                            # http://127.0.0.1:8600  (operator1 / MOCKBANK_PASSWORD)
```

Terminal 2:

```bash
# 1. The agent completes a goal (LLM-driven) and saves a capability
uv run mm discover "Sign in, look up member 10234 and read their current Share Savings balance" \
  --name corebank.demo.savings_balance -p member_id=10234 -o savings_balance --headed

# 2. Replay the resulting artifact with a different input: deterministic, no LLM
uv run mm replay capabilities/corebank.demo.savings_balance/0.1.0.yaml -p member_id=10871

# 3. Error and outcome handling
uv run mm replay capabilities/corebank.demo.savings_balance/0.1.0.yaml -p member_id=99999   # MEMBER_NOT_FOUND
uv run mm replay capabilities/corebank.demo.savings_balance/0.1.0.yaml -p member_id=12ab    # INPUT_INVALID
curl -X POST localhost:8600/__control/faults -H 'content-type: application/json' -d '{"faults":["notice"]}'
uv run mm replay capabilities/corebank.demo.savings_balance/0.1.0.yaml -p member_id=10871   # recovered
curl -X POST localhost:8600/__control/reset
```

The committed capabilities can be replayed directly as well:

```bash
uv run mm replay capabilities/corebank.member.get_savings_balance/0.3.0.yaml -p member_id=10871
# the same capability on another institution running the same product (start it first:
#   uv run mm mockbank --tenant tenant_b --port 8601)
uv run mm replay capabilities/corebank.member.get_savings_balance/0.3.0.yaml -p member_id=10871 \
  --tenant tenant_b --base-url http://127.0.0.1:8601
```

### Irreversible steps and human handoff

`open_sub_account` ends with an irreversible **Confirm**, which only replays under a reviewer's approval:

```bash
O=capabilities/corebank.member.open_sub_account/0.3.0.yaml
uv run mm replay $O -p member_id=10871 -p "account_type=Holiday Club" -p deposit=25 -p nickname=Fund
#   -> POLICY_BLOCKED at s14_click_confirm unless an approval for this exact content exists
uv run mm approve $O --by <your-name>          # writes 0.3.0.approval.yaml, bound to the content hash
```

Handoff: switch on a popup no detector knows, and run with `--escalate`. The run stops and prints an operator
console URL (with a per-run token). Take control there, click **Maybe Later** in the automation's browser
window, press **Resume**, and replay re-verifies the screen and finishes.

```bash
curl -X POST localhost:8600/__control/faults -H 'content-type: application/json' -d '{"faults":["survey"]}'
uv run mm replay $O -p member_id=10871 -p "account_type=Holiday Club" -p deposit=25 -p nickname=Fund --escalate
```

Add `--slow-mo 400` to any replay or discovery to watch it at human speed.

## CLI

| Command | Purpose |
|---|---|
| `mm discover GOAL --name ID [-p k=v] [-o output]` | LLM-driven run; on success compiles and saves `capabilities/ID/<next version>.yaml` |
| `mm replay ARTIFACT [-p k=v] [--tenant T] [--base-url URL]` | deterministic replay; prints the result as JSON |
| `mm approve ARTIFACT --by NAME [--tenant T]` | reviewer sign-off enabling irreversible steps (bound to content, version, tenant) |
| `mm schema` | the capability JSON Schema (the contract a calling agent reads) |
| `mm mockbank [--tenant T] [--faults f1,f2]` | the target application |
| `mm doctor [--ping]` | checks LLM provider configuration |

Common options: `--headed/--headless`, `--escalate`, `--simulate-operator` (a scripted stand-in for a human,
for demos and tests), `--slow-mo MS`, `--trace` (Playwright trace; unredacted, local debugging only).

Replay exit codes: `0` success, `3` business outcome, `2` failure.

Mock bank faults (`--faults`, or `POST /__control/faults` while it runs): `slow`, `notice`, `survey`,
`permission`, `error500`, `session_expired`, `session_expired_on_detail`, `session_expired_on_submit`,
`session_expired_on_confirm`.

## Repository map

```
src/mm/
  agent/        discovery loop, typed decisions, prompts            (the only place a model decides)
  artifact/     schema, compiler, store, approval, packs, tenancy   (the capability and its lifecycle)
  replay/       deterministic executor and the result contract      (the production path)
  surface/      the surface interface + the Playwright web surface  (the perceive/act seam)
  policy/       safety policy and GuardedSurface                    (the single enforcement point)
  handoff/      control lease, interventions, operator console
  evidence/     per-run recorder with redaction
config/policy.yaml    allowlist, action types, risk rules
packs/corebank.yaml   detector pack for the CoreOne product (shared by its capabilities)
tenants/              per-tenant profiles (vocabulary overrides)
capabilities/         recorded capabilities (immutable versions)
mock_bank/            the target application
evidence/             committed runs for review; scripts/make_evidence.py regenerates them
```
