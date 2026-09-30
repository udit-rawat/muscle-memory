# REPORT

I optimised for the production path being correct rather than merely robust: a replay that returns a wrong
answer is worse than one that stops. So the depth went into the capability contract, into separating business
outcomes from failures, and into never guessing, repeating or silently re-targeting an action that could move
money; everything else is thin but real.

## 1. Architecture

One Python process, one module per responsibility, with two hard seams:

- **The only place a model decides is discovery** (`agent/loop.py`). It observes the screen, asks the LLM
  for one typed action, and acts. Its output is a list of recorded steps whose locators were built and
  *verified by code*, not by the model; the transcript is evidence only. The compiler turns that into a
  capability. **Replay** (`replay/executor.py`) never imports an LLM.
- **Everything touches the UI through `GuardedSurface`** (`policy/guard.py`), the single enforcement point for
  the allowlist, action types, the human/automation control lease and irreversible-action authorisation.
  Below it, the `Surface` interface (`surface/base.py`) separates *how we perceive and act* from *the recorded
  flow*; `WebSurface` (Playwright/Chromium) is the one implementation.

Key decisions:

| Decision | Choice | Why / trade-off |
|---|---|---|
| Perception | Accessibility tree (roles, names) + visible text; screenshots only for evidence | Survives markup churn, exists on legacy web and desktop; a pixel-only agent cannot produce replayable locators. Loses canvas-only UIs (see Cuts). |
| LLM | Groq `gpt-oss-120b` (free tier), then a list of Gemini models as fallbacks, all via the OpenAI-compatible API; Instructor for typed output | Cheap and fast enough for discovery. Failover is on the root cause (outage, rate limit, invalid output fail over; a bad key does not), and walks the list: free-tier models are often "high demand", so one fallback is not enough. Proven with a full discovery run with Groq unreachable. |
| Enforcement of agent behaviour | Pydantic validators bound to the current screen, not prompt instructions | Model mistakes seen in real runs (skipping an input, re-reading a value, reading through a modal) were fixed by *rejecting* invalid decisions, which Instructor re-asks. |
| Frameworks | No LangGraph/LangChain/LiteLLM | The loop is small; the valuable part (pause, cede, resume on the same browser) is not a graph checkpoint. LiteLLM was avoided after its 2026 supply-chain compromise; two OpenAI-compatible providers need only a small router. |
| Storage | YAML files, immutable versions, git-reviewable | Artifacts are read by humans in a diff; no database needed at this scale. |
| Target | A local mock core-banking app with injectable faults | Real banks are off-limits; faults (expiry, popups, 500s, permission) are what replay must handle, and they must be reproducible. |

**Why these tools.** Python because the typed-LLM tooling (Pydantic, Instructor) and the test tooling are
strongest there, and one language for the agent, compiler and mock app kept the system small. Playwright
because it gives the accessibility tree, frames, network interception and tracing in one library. A pixel-
based computer-use SDK decides well but records coordinates, which don't replay deterministically, and it
would hide the locator strategy I wanted to own. TypeScript with Zod would have been an equally valid choice.

## 2. Artifact schema

A capability (`artifact/schema.py`, JSON Schema via `mm schema`) is a **contract** plus a **procedure** plus
**knowledge of runtime states**:

- **Contract**: `id` (dotted), `version` (semver, immutable once saved), `inputs` (typed; patterns, enums
  inferred from dropdowns, `sensitive`), `outputs` (typed, `parse: currency`, masked in logs), `outcomes`
  (the business codes a caller may receive), `secrets` (names only, e.g. `{{secret:MOCKBANK_PASSWORD}}`),
  `app` (name, entry path, never a host), `provenance` (discovery run, model, tenant recorded on).
- **Procedure**: `steps`, each with `intent` (for reviewers), `action`, `target`, `value` template,
  `expect` checkpoints, `risk` (`safe` / `mutating` / `irreversible`), `timeout_ms`.
- **Targets** are an ordered list of strategies, most semantic first: `role`+name, `title`, `text`,
  `table_cell` (*row whose key is "Share Savings" × column "Balance"*, never the value itself), then
  structural `attr`/`css`. Every strategy must match **exactly one** element at record time (verified
  against the live element) and at replay time; ambiguity is a miss, never a guess.
- **Detectors**: declarative runtime states with a class (`business_outcome` / `recoverable` /
  `hard_failure`), trigger conditions, and for recoverable ones a bounded, deterministic handler
  (`then: continue | retry_step | restart`, `max_times`). They come from an app-level **pack**
  (`packs/corebank.yaml`, written once per vendor product) and from **discovery**: a popup the agent had to
  dismiss becomes a recoverable detector, not a step (it will not be there on every run).

Why this shape: a calling agent needs the header only (what it needs, what it gets, which answers exist);
a reviewer needs to read the procedure and the error model separately; and the error model belongs to the
application, not to each flow, hence packs. The schema validates its own consistency (every referenced
input, secret, output and outcome is declared, and vice versa).

## 3. Determinism & error handling

**Determinism.** Same artifact + same inputs + same application state → same steps and outputs. There are
no fixed sleeps: after each action replay waits until the network is quiet, then until the step's
checkpoints hold (the page URL it must land on, or the next control being present). Evidence scenario 17:
20/20 replays, 0 drift.

**Runtime states.** While waiting, detectors are checked on every poll, and again whenever a step fails, so
a state is recognised for what it is: "no *Select* link" after a search is `MEMBER_NOT_FOUND`, not
`TARGET_NOT_FOUND`. The result contract separates:

- `success` (outputs, plus every `recovery` that happened: a dismissed notice, a safe restart after a
  session expiry),
- `business_outcome` (a declared code with the application's own message: a legitimate answer),
- `failure` (a kind, the step, what was expected, what was observed, a masked screenshot, and
  `may_have_committed`).

Rules that make this safe rather than merely robust (each found by auditing the system and now tested):

- **Reads never use positional locators.** A CSS path to "row 2, column 4" once returned a *checking*
  balance for a member without savings; reads now resolve only semantically, and a miss is a failure.
- **Positional locators may find, but not decide.** An action proceeds on a CSS/attribute match only if the
  element reads as the expected control; on tenant B's reordered menu the old path pointed at "Dashboard"
  and is refused (evidence 15).
- **Nothing that may have committed is repeated.** Once a `mutating` or `irreversible` step is dispatched,
  no retry, redo or restart may pass it: `UNSAFE_TO_REPEAT` (evidence 14).
- **Nothing unrecognised is ignored.** A blocking overlay that no detector handles is `UNEXPECTED_STATE`;
  replay never carries on underneath it.
- Unreachable app, closed browser, unexpected exceptions: `APP_UNREACHABLE`, `SESSION_CLOSED`,
  `INTERNAL_ERROR`, never a traceback.

**Drift.** Every step reports which strategy matched; a fallback match is reported as `drift`, the signal
for re-review (see §4).

## 4. Heterogeneity & multi-tenant

**Other surfaces.** The artifact holds no Playwright code: targets are roles, names, titles and table
anchors, which is also the vocabulary of desktop accessibility APIs (Windows UI Automation, macOS AX). A
`DesktopSurface` would implement the same `Surface` protocol (`observe`, `perform`, `check`,
`blocking_overlays`) against the accessibility tree; a screenshot crop captured at record time (not recorded
today) would be the last-resort visual fallback. Legacy web is already the implemented case: framesets are addressed by
`frame_path`, and tables by header/row anchors.

**Multi-tenant reuse.** A capability is keyed by vendor product (its pack declares `applies_to`), not by
tenant, and is recorded once. Per-tenant differences live in a small **tenant profile**
(`tenants/tenant_b.yaml`): exact renames of on-screen vocabulary, applied to every capability of that app,
plus optional per-step patches for structural differences. Evidence 15/16: the capability recorded on
tenant A fails cleanly on tenant B (renamed labels, swapped menu), then succeeds with a 7-line profile and no
re-recording. A specialised capability is distinct content, so its irreversible steps need their own
approval (`mm approve --tenant`).

**Drift management at scale** (designed, not built): fingerprint each tenant's app version and check it
against the pack's `applies_to`; aggregate per-tenant strategy telemetry (fallback hits, misses); run a
canary replay per tenant and version, and on failure mark the capability `needs_review` for that tenant
only, instead of breaking every tenant.

**Operating it.** I'd run a central catalogue: capabilities per vendor product version, tenant profiles as
overlays, and replay telemetry per tenant. Fallback-strategy hits and failed checkpoints feed a drift
dashboard; past a threshold the capability is marked needs_review for that tenant, and the calling agent gets
a clear 'unavailable' instead of a risky run.

## 5. Escalation & handoff

**Detecting "stuck".** Discovery: the agent asks for help, makes no progress three times, or wants to click
an irreversible control. Replay: an unrecoverable state (unknown overlay, missing target, failed checkpoint,
exhausted recovery). Business outcomes never escalate; they are answers.

**Routing.** An intervention request (`handoff/intervention.py`) carries the capability or goal, the step,
the reason, expected vs observed state, a masked screenshot and a masked page excerpt. It is saved to the
run's evidence and shown in the operator console (`handoff/console.py`), which requires a per-run token and
refuses cross-origin requests.

**Control transfer.** A `ControlLease` per live session: `AGENT` / `HUMAN` / `NONE` plus an epoch that every
transfer increments. `GuardedSurface` checks it before every automated action, so while a human holds the
session automation cannot act, and after the human hands back, automation stays locked out until it has
re-verified the screen (`resync`). The human works in the **same browser session** (headed); their clicks
and field changes are captured (never typed values) and recorded against the intervention.

**Hand-back.** Replay re-checks overlays and detectors, then the step's checkpoints: if the human completed
the step it moves on, if the step is safe it redoes it, and if it may already have committed it stops
(`UNSAFE_TO_REPEAT`). A discovery run in which a human took control does not compile, since their actions
are not replayable steps yet. Evidence 13 shows the full cycle.

Mocked deliberately: the console is a minimal page, and the operator in automated tests and evidence is a
scripted stand-in (`--simulate-operator`) that uses the real console API and real clicks in the live
session, and is labelled as simulated everywhere. With a real person, `--escalate` opens a visible browser.

## 6. Safety

- **Allowlist.** `config/policy.yaml` is bound per run to the tenant's exact origin; the operator console
  and denied paths (`/__control`, `/logout`) are refused. It is enforced twice: before each action by
  `GuardedSurface`, and on **every network request** from any frame by a browser-level filter, so page
  scripts and redirects are covered too.
- **Risk.** Three levels from the policy, judged on a control's accessible name: `irreversible` (Confirm,
  Post, Transfer…), `mutating` (Continue, Save, Submit…), `safe`. Irreversible actions pause for a human's
  approval during discovery, and at replay require an **approval** bound to the capability id, version,
  tenant and a hash of its content, checked at the moment of use. Commit requests (e.g. `POST …_confirm.jsp`)
  are also blocked at the network layer outside an authorised step. Mutating steps are never repeated.
- **Credentials.** Fields that ask for a credential (password, user/operator id, PIN) only accept a
  `{{secret:NAME}}` placeholder: a fallback model was seen typing a *guessed* login ("admin"), which in a
  bank could lock an operator out; it is now rejected and re-asked, enforced in code, not in the prompt.
- **Data.** Secrets are never literal (`{{secret:NAME}}`), never logged (raw and JSON-escaped forms
  scrubbed). Regulated values are masked in the prompts sent to the model (which extracts by pointing at an
  element, so it never needs them), in logs, screenshots, intervention records and evidence; the prompt
  logged is byte-identical to the prompt sent. Playwright traces cannot be redacted, so they are opt-in and
  never part of evidence. A test scans every committed evidence byte for secrets and seeded values.

**Limits.** Masking is pattern-based (amounts, SSNs, account numbers, emails, phones); names are not caught.
Approvals prove integrity, not identity: they are not signed, and the approver's name is self-declared (a
production system would issue approvals from an authenticated review service). Risk classification by
control name is coarse, and the network-level commit rule is per application. During discovery the model
sees masked screen text; production would additionally need a zero-retention provider agreement.

## 7. Cuts

Left out deliberately, and what I would build next:

1. **Human actions compiled into steps**: capture the human's element as a verified target during a
   takeover, so assisted runs can compile (today they are refused).
2. **Bounded LLM fallback for one step** on replay failure (the step `intent` is already stored for it),
   policy-checked and recorded as evidence.
3. **Desktop surface** against Windows UI Automation, and a remote operator view (CDP screencast) instead of a
   local headed window.
4. **Drift operations**: version fingerprinting, per-tenant canaries and telemetry (designed in §4).
5. **Signed approvals and authenticated operators** (SSO), plus an agent-facing capability catalogue.
6. **Richer PII detection** (names, addresses) and native dialogs (`alert`/`confirm`), which the mock app
   does not use.

**What I'd do first.** First, compile the steps a human performs during a takeover, so every escalation makes
the capability better instead of just unblocking it. Second, signed approvals from an authenticated review
service, since integrity without identity is the weakest part of the safety story. What I'd do differently:
start the regression-test-first audit loop on day one. Three audits found real bugs (a read returning another
account's balance, an approval authorising a different capability) that earlier adversarial tests would have
caught.
