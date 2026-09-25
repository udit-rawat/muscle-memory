"""Web surface on Playwright/Chromium, perceived through roles and accessible names.

Discovery: `observe()` enumerates actionable elements across all frames and hands the LLM short
refs ("e12"). When the LLM acts on a ref, we build a *locator bundle* for that element and keep
only strategies verified to match exactly that element, right now, on the live page.

Replay: `perform()` resolves a recorded Target by trying its strategies in order; every
strategy must match exactly one element (ambiguity is a miss, never a guess).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import ElementHandle, Frame, Locator, Page, sync_playwright
from playwright.sync_api import Error as PlaywrightError

from mm.surface import js
from mm.surface.base import (
    ActionType,
    ActResult,
    AttrStrategy,
    Checkpoint,
    CssStrategy,
    ElementRef,
    FrameText,
    Observation,
    ResolveResult,
    RoleStrategy,
    Strategy,
    TableCellStrategy,
    Target,
    TextStrategy,
    TitleStrategy,
)

MAX_ELEMENTS = 150
FRAME_TEXT_CHARS = 700
_POLL_MS = 150


class WebSurface:
    def __init__(self, headless: bool = True, trace_path: Path | None = None, slow_mo_ms: int = 0) -> None:
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless, slow_mo=slow_mo_ms)
        self._context = self._browser.new_context(viewport={"width": 1280, "height": 800})
        self._trace_path = trace_path
        if trace_path:
            self._context.tracing.start(screenshots=True, snapshots=True)
        self.page: Page = self._context.new_page()
        self._inflight = 0
        self.page.on("request", self._on_request)
        self.page.on("requestfinished", self._on_request_done)
        self.page.on("requestfailed", self._on_request_done)
        self._refs: dict[str, tuple[Frame, ElementHandle, dict[str, Any]]] = {}

    # --- lifecycle -----------------------------------------------------------------------------

    def close(self) -> None:
        try:
            if self._trace_path:
                self._context.tracing.stop(path=str(self._trace_path))
            self._context.close()
            self._browser.close()
        finally:
            self._pw.stop()

    def screenshot(self, path: str) -> None:
        self.page.screenshot(path=path)

    def navigate(self, url: str) -> None:
        self.page.goto(url, wait_until="domcontentloaded")
        self._settle()

    # --- discovery -----------------------------------------------------------------------------

    def observe(self, with_screenshot: bool = False) -> Observation:
        self._refs.clear()
        elements: list[ElementRef] = []
        frames: list[FrameText] = []
        for frame in self.page.frames:
            path = frame_path(frame)
            try:
                text = str(frame.evaluate(js.BODY_TEXT))
            except PlaywrightError:
                continue  # frame navigated away mid-observation; next observe will catch it
            if text.strip():
                frames.append(FrameText(frame_path=path, url=frame.url, text=_squash(text)[:FRAME_TEXT_CHARS]))
            for selector in (js.INTERACTIVE_SELECTOR, js.READABLE_SELECTOR):
                for handle in frame.query_selector_all(selector):
                    if len(elements) >= MAX_ELEMENTS:
                        break
                    desc = handle.evaluate(js.DESCRIBE)
                    if not desc["visible"] or (desc["role"] == "cell" and (desc["nested"] or not desc["name"])):
                        continue
                    ref = f"e{len(elements) + 1}"
                    self._refs[ref] = (frame, handle, desc)
                    elements.append(ElementRef(ref=ref, role=desc["role"], name=desc["name"][:80],
                                               value=desc["value"], options=desc["options"], frame_path=path))
        shot = self.page.screenshot() if with_screenshot else None
        return Observation(url=self.page.url, title=self.page.title(), elements=elements,
                           frames=frames, screenshot_png=shot)

    def act(self, action: ActionType, ref: str | None, value: str | None = None) -> ActResult:
        if action is ActionType.NAVIGATE:
            if not value:
                return ActResult(ok=False, detail="navigate needs a url")
            self.navigate(value)
            return ActResult(ok=True, frame_urls=self.frame_urls())
        if ref not in self._refs:
            return ActResult(ok=False, detail=f"unknown ref {ref!r}; use a ref from the latest observation")
        frame, handle, desc = self._refs[ref]
        target = self._build_target(frame, handle, desc, action)
        try:
            extracted = self._do(action, handle, value)
        except PlaywrightError as exc:
            return ActResult(ok=False, detail=_first_line(exc), target=target)
        self._settle()
        return ActResult(ok=True, target=target, extracted=extracted, options=desc["options"],
                         frame_urls=self.frame_urls())

    # --- replay --------------------------------------------------------------------------------

    def resolve(self, target: Target, timeout_ms: int) -> ResolveResult:
        found, _, idx, attempts = self._resolve(target, timeout_ms)
        return ResolveResult(found=found is not None, strategy_index=idx, attempts=attempts)

    def perform(self, action: ActionType, target: Target | None, value: str | None, timeout_ms: int) -> ActResult:
        if action is ActionType.NAVIGATE:
            return self.act(action, None, value)
        if target is None:
            return ActResult(ok=False, detail=f"{action} needs a target")
        loc, _, idx, attempts = self._resolve(target, timeout_ms)
        if loc is None:
            return ActResult(ok=False, detail="target not found", attempts=attempts)
        handle = loc.element_handle(timeout=timeout_ms)
        try:
            extracted = self._do(action, handle, value)
        except PlaywrightError as exc:
            return ActResult(ok=False, detail=_first_line(exc), strategy_index=idx, attempts=attempts)
        self._settle()
        return ActResult(ok=True, strategy_index=idx, attempts=attempts, extracted=extracted,
                         frame_urls=self.frame_urls())

    def check(self, checkpoint: Checkpoint, timeout_ms: int) -> tuple[bool, str]:
        """Poll until the condition holds or `timeout_ms` passes (0 = check once). Returns (held, observed)."""
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            held, observed = self._check_once(checkpoint)
            if held or time.monotonic() >= deadline:
                return held, observed
            self.page.wait_for_timeout(_POLL_MS)

    def blocking_overlays(self) -> list[str]:
        """Visible fixed-position layers covering a large part of a frame: modal dialogs, lightboxes.

        Detectors handle the overlays we know about; any other one means the UI is in a state the
        capability was never recorded in, and replay must not carry on underneath it.
        """
        found: list[str] = []
        for frame in self.page.frames:
            try:
                text = frame.evaluate(js.BLOCKING_OVERLAY)
            except PlaywrightError:
                continue
            if text is not None:
                where = "/".join(frame_path(frame)) or "top"
                found.append(f"overlay in frame {where}: {_squash(str(text))[:120]!r}")
        return found

    def _check_once(self, cp: Checkpoint) -> tuple[bool, str]:
        frames = self.page.frames if cp.any_frame else [f for f in [self._frame(cp.frame_path)] if f]
        if not frames:
            return False, f"frame {'/'.join(cp.frame_path)} not found"
        observed: list[str] = []
        for frame in frames:
            try:
                if cp.kind == "url_matches":
                    if re.search(cp.pattern or "", frame.url):
                        return True, frame.url
                    observed.append(frame.url)
                elif cp.kind in ("text_visible", "text_matches"):
                    body = str(frame.evaluate(js.BODY_TEXT))
                    if cp.kind == "text_visible" and (cp.pattern or "") in _squash(body):
                        line = next((ln for ln in body.splitlines() if (cp.pattern or "") in _squash(ln)), "")
                        return True, _squash(line)[:200] or (cp.pattern or "")
                    if cp.kind == "text_matches":
                        for line in body.splitlines():
                            if re.search(cp.pattern or "", line):
                                return True, _squash(line)[:200]
                    observed.append(_squash(body)[:150])
                elif cp.kind == "target_present" and cp.target is not None:
                    target = cp.target if not cp.any_frame else cp.target.model_copy(
                        update={"frame_path": frame_path(frame)})
                    loc, *_ = self._resolve(target, 0)
                    if loc is not None:
                        return True, "present"
                    observed.append("absent")
            except PlaywrightError as exc:  # frame navigated mid-check: treat as not (yet) holding
                observed.append(_first_line(exc))
        return False, " | ".join(o for o in observed if o)[:300]

    def frame_urls(self) -> dict[str, str]:
        return {"/".join(frame_path(f)): f.url for f in self.page.frames}

    # --- internals -----------------------------------------------------------------------------

    def _do(self, action: ActionType, handle: ElementHandle, value: str | None) -> str | None:
        if action is ActionType.CLICK:
            handle.click(timeout=5000)
        elif action is ActionType.FILL:
            handle.fill(value or "", timeout=5000)
        elif action is ActionType.SELECT:
            try:
                handle.select_option(label=value or "", timeout=5000)
            except PlaywrightError:
                handle.select_option(value=value or "", timeout=5000)
        elif action is ActionType.EXTRACT:
            return _squash(handle.inner_text())
        return None

    def _build_target(self, frame: Frame, handle: ElementHandle, desc: dict[str, Any], action: ActionType) -> Target:
        """Candidate strategies, most semantic first; keep only those that uniquely hit *this* element."""
        candidates: list[Strategy] = []
        if action is ActionType.EXTRACT and desc["tag"] == "td":
            ctx = handle.evaluate(js.CELL_CONTEXT)
            header = ctx["headers"][ctx["idx"]] if ctx["headers"] and ctx["idx"] < len(ctx["headers"]) else None
            # Prefer word-like keys ("Share Savings") over codes ("S00"): closer to how a human names the row.
            for key in sorted(set(ctx["keys"]), key=lambda k: (any(c.isdigit() for c in k), -len(k))):
                candidates.append(TableCellStrategy(row_key=key, column_header=header,
                                                    column_index=None if header else ctx["idx"] + 1))
        else:
            if desc["role"] != "generic" and desc["name"]:
                candidates.append(RoleStrategy(role=desc["role"], name=desc["name"]))
            if desc["title"]:
                candidates.append(TitleStrategy(title=desc["title"]))
            if desc["tag"] in ("a", "button") and desc["text"]:
                candidates.append(TextStrategy(text=desc["text"]))
            if desc["name_attr"]:
                candidates.append(AttrStrategy(tag=desc["tag"], name=desc["name_attr"]))
        candidates.append(CssStrategy(selector=desc["css"]))

        verified = [s for s in candidates if self._is_unique_match(frame, s, handle)]
        path = frame_path(frame)
        where = f" in frame {'/'.join(path)}" if path else ""
        first = verified[0] if verified else None
        if isinstance(first, TableCellStrategy):
            # Never describe a read by its content: that would persist the value (e.g. a balance) in the artifact.
            column = first.column_header or f"column {first.column_index}"
            label = f'cell [row "{first.row_key}" x {column}]'
        else:
            label = f'{desc["role"]} "{desc["name"][:60]}"'
        return Target(frame_path=path, strategies=verified or candidates[-1:], description=label + where)

    def _is_unique_match(self, frame: Frame, strategy: Strategy, handle: ElementHandle) -> bool:
        try:
            loc = locator_for(frame, strategy)
            return loc.count() == 1 and bool(loc.first.evaluate("(el, h) => el === h", handle))
        except PlaywrightError:
            return False

    def _resolve(self, target: Target, timeout_ms: int) -> tuple[Locator | None, Frame | None, int | None, list[str]]:
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            attempts: list[str] = []
            frame = self._frame(target.frame_path)
            if frame is None:
                attempts.append(f"frame {'/'.join(target.frame_path)!r} not found")
            else:
                for i, strategy in enumerate(target.strategies):
                    try:
                        loc = locator_for(frame, strategy)
                        n = loc.count()
                    except PlaywrightError as exc:
                        attempts.append(f"{strategy.by}: error {_first_line(exc)}")
                        continue
                    if n == 1:
                        return loc, frame, i, attempts
                    attempts.append(f"{strategy.by}: {n} matches")
            if time.monotonic() >= deadline:
                return None, frame, None, attempts
            self.page.wait_for_timeout(_POLL_MS)

    def _frame(self, path: list[str]) -> Frame | None:
        frame = self.page.main_frame
        for part in path:
            children = frame.child_frames
            nxt = next((c for c in children if c.name == part), None)
            if nxt is None and part.startswith("#") and part[1:].isdigit() and int(part[1:]) < len(children):
                nxt = children[int(part[1:])]
            if nxt is None:
                return None
            frame = nxt
        return frame

    def _settle(self, quiet_ms: int = 250, timeout_ms: int = 15000) -> None:
        """Wait until no request has been in flight for `quiet_ms` (across all frames)."""
        deadline = time.monotonic() + timeout_ms / 1000
        quiet_since = time.monotonic()
        while time.monotonic() < deadline:
            self.page.wait_for_timeout(50)  # also pumps Playwright events that update _inflight
            if self._inflight > 0:
                quiet_since = time.monotonic()
            elif (time.monotonic() - quiet_since) * 1000 >= quiet_ms:
                return

    def _on_request(self, _: object) -> None:
        self._inflight += 1

    def _on_request_done(self, _: object) -> None:
        self._inflight = max(0, self._inflight - 1)


def frame_path(frame: Frame) -> list[str]:
    path: list[str] = []
    while frame.parent_frame is not None:
        parent = frame.parent_frame
        path.insert(0, frame.name or f"#{parent.child_frames.index(frame)}")
        frame = parent
    return path


def locator_for(frame: Frame, s: Strategy) -> Locator:
    if isinstance(s, RoleStrategy):
        return frame.get_by_role(s.role, name=s.name, exact=True)  # type: ignore[arg-type]
    if isinstance(s, TitleStrategy):
        return frame.get_by_title(s.title, exact=True)
    if isinstance(s, TextStrategy):
        return frame.get_by_text(s.text, exact=True)
    if isinstance(s, AttrStrategy):
        return frame.locator(f'{s.tag}[name="{s.name}"]')
    if isinstance(s, TableCellStrategy):
        return frame.locator("xpath=" + table_cell_xpath(s))
    return frame.locator(s.selector)


def table_cell_xpath(s: TableCellStrategy) -> str:
    row = f"//tr[td[normalize-space(.)={xpath_literal(s.row_key)}]]"
    if s.column_header is not None:
        h = xpath_literal(s.column_header)
        col = f"count(ancestor::table[1]//tr[th][1]/th[normalize-space(.)={h}]/preceding-sibling::*) + 1"
        return f"{row}/*[position() = {col}]"
    return f"{row}/*[{s.column_index or 1}]"


def xpath_literal(s: str) -> str:
    if "'" not in s:
        return f"'{s}'"
    if '"' not in s:
        return f'"{s}"'
    return "concat(" + ", \"'\", ".join(f"'{p}'" for p in s.split("'")) + ")"


def _squash(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", text)).strip()


def _first_line(exc: Exception) -> str:
    return str(exc).strip().splitlines()[0][:200] if str(exc).strip() else type(exc).__name__
