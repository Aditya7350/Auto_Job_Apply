"""Application sessions.

Replaces the old `await page.wait_for_timeout(300000)` + close-the-browser pattern.

Lifecycle of one session:
    starting -> opening -> (login_required) -> filling -> [Next, fill, Next, fill...] ->
        ready_for_review  (stopped on a page: a field needs the user, or no Next button)
        ready_to_submit   (every page done; the form shows only its final Submit button)
    ready_for_review --refresh/refill/next--> ready_for_review | ready_to_submit
    ready_for_review | ready_to_submit --approve--> submitting -> submitted | submitted_unverified
    any state --cancel--> cancelled          (idle too long -> timeout)

The browser stays open for as long as the session lives, so the user can review
and edit directly in the real browser window. NOTHING is submitted except through
the `approve` command, which the review page only sends when the user clicks it.
"""
from __future__ import annotations

import asyncio
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from playwright.async_api import async_playwright

from app.accounts import click_by_label
from app.navigator import dismiss_overlays
from app.pipeline import retype_rejected_values, snapshot

log = logging.getLogger("auto_job_apply")

NEXT_RX = re.compile(r"^\s*(next|next step|next page|continue|save (and|&) (continue|next)|proceed|"
                     r"continue to (the )?(next step|review)|review (my )?application)\s*$", re.I)
SUBMIT_RX = re.compile(r"^\s*(submit( application)?|send application|complete application|apply( now)?)\s*$", re.I)
# The button that ends the application. Unlike SUBMIT_RX no "Apply now": that one opens a form, it doesn't end it.
FINAL_SUBMIT_RX = re.compile(r"^\s*(submit( (my )?application)?|send( my)? application|complete( my)? application|"
                             r"finish( application)?)\s*$", re.I)
MAX_STEPS = 15
BUTTONS = 'button, input[type="submit"], input[type="button"], [role="button"]'

# A visible loading indicator: spinner, progress bar, aria-busy region, Workday's loading page.
BUSY_JS = r"""
() => [...document.querySelectorAll('[aria-busy="true"], [role="progressbar"], [class*="spinner" i], ' +
       '[class*="loading" i], [class*="loader" i], [data-automation-id*="loading" i]')]
  .some(e => { const r = e.getBoundingClientRect(), st = getComputedStyle(e);
               return r.width > 0 && r.height > 0 && st.visibility !== 'hidden' && st.display !== 'none' && +st.opacity > 0.05; })
"""

# Name of the step the form is on ("My Experience"), from the progress bar when the site marks it.
STEP_NAME_JS = r"""
() => {
  const n = document.querySelector('[aria-current="step"], [aria-current="true"], [data-automation-id="progressBarActiveStep"]');
  const t = n ? (n.innerText || n.getAttribute('aria-label') || '') : '';
  return t.replace(/\s+/g, ' ').trim().slice(0, 80);
}
"""
SUCCESS_RX = re.compile(r"thank you|application (has been |was )?(submitted|received|sent)|"
                        r"successfully (submitted|applied)|we('ve| have) received your", re.I)
TERMINAL = {"submitted", "submitted_unverified", "cancelled", "timeout", "error"}


@dataclass
class Command:
    name: str
    args: dict
    future: asyncio.Future


@dataclass
class Session:
    id: int
    url: str
    token: str = field(default_factory=lambda: secrets.token_urlsafe(16))
    status: str = "starting"
    message: str = ""
    title: str = ""
    rows: list = field(default_factory=list)  # what the engine filled (from fill_application)
    live_rows: list = field(default_factory=list)  # what the form contains now (from snapshot)
    summary: dict = field(default_factory=dict)
    required_empty: list = field(default_factory=list)
    screenshot: str = ""
    created_at: float = field(default_factory=time.time)
    filled_total: int = 0  # cumulative across steps/re-scans
    step: int = 1  # page of a multi-step form the automation is on
    step_name: str = ""
    final_reached: bool = False  # on the last page: only the Submit button is left
    review_text: str = ""  # what the site's own final Review page shows
    blockers: list = field(default_factory=list)  # "Email address: Please enter email address."
    pages: list = field(default_factory=list)
    save_answers: Optional[Callable[[list], int]] = None
    busy: bool = False
    loop_ready: bool = False  # review loop is accepting commands
    finished: bool = False
    # runtime objects (never serialised)
    pw: Any = None
    browser: Any = None
    context: Any = None
    page: Any = None
    refill: Optional[Callable[[], Awaitable[dict]]] = None  # set by the runner
    commands: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: Optional[asyncio.Task] = None
    persist: Optional[Callable[["Session"], None]] = None

    def set(self, status: str, message: str = "") -> None:
        self.status, self.message = status, message
        log.info("SESSION %s | %s | %s", self.id, status, message)
        if self.persist:
            try:
                self.persist(self)
            except Exception as exc:  # DB problems must never kill the browser session
                log.warning("persist failed: %s", exc)

    def apply_result(self, result: dict, merge: bool = False) -> None:
        """Store a fill_application() result. With merge=True (a re-scan of the same page) rows the
        automation filled earlier keep their source/confidence instead of turning into 'kept'."""
        rows = result.get("rows", [])
        for r in rows:
            r.setdefault("step", self.step)
            r.setdefault("step_name", self.step_name)
        if merge and self.rows:
            old = {r["label"]: r for r in self.rows if r.get("label") and r.get("status") == "filled"}
            rows = [old[r["label"]] if r.get("status") == "kept" and r.get("label") in old else r for r in rows]
        self.rows = rows
        self.filled_total += result.get("filled", 0)
        self.summary = {"filled": self.filled_total,
                        "needs_review": sum(1 for r in rows if r.get("needs_review"))}
        self.required_empty = result.get("required_unfilled", [])

    def public(self) -> dict:
        return {
            "id": self.id, "url": self.url, "title": self.title, "status": self.status, "message": self.message,
            "rows": self.rows, "live_rows": self.live_rows, "summary": self.summary,
            "required_empty": self.required_empty, "screenshot": self.screenshot, "busy": self.busy,
            "step": self.step, "final_reached": self.final_reached, "review_text": self.review_text,
            "blockers": self.blockers,
            "pages": self.pages,
            "missing": [] if self.finished else list(self.required_empty),
            "ready": self.loop_ready and not self.finished, "finished": self.finished,
        }


Runner = Callable[[Session], Awaitable[None]]


class SessionManager:
    @staticmethod
    async def _close_browser(s: Session) -> None:
        async def _close():
            for obj, name in ((s.context, "close"), (s.browser, "close"), (s.pw, "stop")):
                if obj is None:
                    continue
                try:
                    await asyncio.wait_for(getattr(obj, name)(), timeout=8)
                except BaseException:
                    pass
        t = asyncio.create_task(_close())
        try:
            await asyncio.shield(t)            # a second cancel must not abandon the close
        except asyncio.CancelledError:
            await asyncio.wait({t}, timeout=12)
    def __init__(self, runner: Runner, persist: Optional[Callable[[Session], None]] = None,
                 screenshot_fn: Optional[Callable[[Any, str], Awaitable[str]]] = None,
                 max_active: int = 3, headless: bool = False,
                 review_timeout_s: int = 1800, post_submit_s: int = 600):
        self.runner, self.persist, self.screenshot_fn = runner, persist, screenshot_fn
        self.max_active, self.headless = max_active, headless
        self.review_timeout_s, self.post_submit_s = review_timeout_s, post_submit_s
        self.sessions: dict[int, Session] = {}

    # ------------------------------------------------------------------ public API
    def get(self, session_id: int) -> Optional[Session]:
        return self.sessions.get(session_id)

    def active_count(self) -> int:
        return sum(1 for s in self.sessions.values() if s.task and not s.task.done())

    async def start(self, url: str, session_id: int) -> Session:
        if self.active_count() >= self.max_active:
            raise RuntimeError(f"Too many open applications (max {self.max_active}). Finish or cancel one first.")
        s = Session(id=session_id, url=url, persist=self.persist)
        self.sessions[session_id] = s
        s.task = asyncio.create_task(self._run(s))
        return s

    async def send(self, s: Session, name: str, **args) -> dict:
        """Send a command to the review loop and wait for its result."""
        if name == "cancel" and not s.loop_ready:
            if s.task and not s.task.done():
                s.task.cancel()  # still opening / waiting for login: stop right away
            return {"ok": True, "message": "Cancelled"}
        if not s.loop_ready or s.task is None or s.task.done():
            return {"ok": False, "message": "The application is still being prepared (or already closed)."}
        if s.finished and name not in {"cancel", "close"}:
            return {"ok": False, "message": "This application is finished."}
        if s.busy:
            return {"ok": False, "message": "Busy with the previous action; try again in a moment."}
        fut = asyncio.get_running_loop().create_future()
        s.commands.put_nowait(Command(name, args, fut))
        try:
            return await asyncio.wait_for(fut, timeout=180)
        except asyncio.TimeoutError:
            return {"ok": False, "message": "The action is taking long; check the browser window."}

    # async def shutdown(self) -> None:
    #     for s in list(self.sessions.values()):
    #         if s.task and not s.task.done():
    #             s.task.cancel()
    #     await asyncio.gather(*(s.task for s in self.sessions.values() if s.task), return_exceptions=True)
    async def shutdown(self) -> None:
        tasks = [s.task for s in self.sessions.values() if s.task and not s.task.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=20)

    # ------------------------------------------------------------------ internals
    async def _run(self, s: Session) -> None:
        try:
            s.pw = await async_playwright().start()
            s.browser = await s.pw.chromium.launch(headless=self.headless)
            s.context = await s.browser.new_context(viewport={"width": 1440, "height": 900})
            s.page = await s.context.new_page()
            await self.runner(s)
            await self._review_loop(s)
        except asyncio.CancelledError:
            s.set("cancelled", "Cancelled by user")
        except Exception as exc:
            log.exception("session %s crashed", s.id)
            s.set("error", f"{type(exc).__name__}: {str(exc)[:300]}")
        finally:
            s.loop_ready, s.finished = False, True
            while not s.commands.empty():  # nobody should be left waiting
                cmd = s.commands.get_nowait()
                if not cmd.future.done():
                    cmd.future.set_result({"ok": False, "message": "Session closed."})
            await self._close_browser(s) 
            # for closer in (getattr(s.browser, "close", None), getattr(s.pw, "stop", None)):
            #     if closer:
            #         try:
            #             await asyncio.wait_for(closer(), timeout=10)
            #         except BaseException:  # includes timeouts and a second cancellation
            #             pass

    async def _review_loop(self, s: Session) -> None:
        s.loop_ready = True
        while True:
            timeout = self.post_submit_s if s.finished else self.review_timeout_s
            try:
                cmd = await asyncio.wait_for(s.commands.get(), timeout=timeout)
            except asyncio.TimeoutError:
                if not s.finished:
                    s.set("timeout", f"No action for {timeout // 60} minutes; browser closed.")
                return
            s.busy = True
            try:
                result = await self._handle(s, cmd)
            except Exception as exc:
                log.exception("command %s failed", cmd.name)
                result = {"ok": False, "message": f"{type(exc).__name__}: {str(exc)[:200]}"}
            finally:
                s.busy = False
            if not cmd.future.done():
                cmd.future.set_result(result)
            if result.get("exit"):
                return

    async def _handle(self, s: Session, cmd: Command) -> dict:
        page = s.page
        if cmd.name in {"cancel", "close"}:
            if not s.finished:
                s.set("cancelled", "Cancelled by user")
            return {"ok": True, "exit": True}

        if cmd.name == "refresh":
            await self._refresh(s)
            return {"ok": True}

        if cmd.name == "refill":  # user fixed something in the browser (clicked Apply, logged in...): re-scan
            s.set("filling", "Re-scanning the page...")
            s.apply_result(await s.refill(), merge=True)
            await self._refresh(s)
            if s.rows:
                s.set("ready_for_review", "Re-scanned. Review the highlighted fields.")
            else:
                s.set("no_application_form", "No application fields found on this page. Go to the form in the "
                                             "browser, then press 'Re-scan & fill'.")
            return {"ok": True}

        if cmd.name == "next":
            outcome = await self._advance_steps(s)
            s.set(outcome["status"], outcome["message"])
            return {"ok": outcome["advanced"] > 0 or s.final_reached, "message": outcome["message"]}
        
        if cmd.name == "answer":
            if s.save_answers:
                s.save_answers(cmd.args.get("answers") or [])
            s.set("filling", "Saved your answers. Filling again...")
            s.apply_result(await s.refill(), merge=True)
            outcome = await self._advance_steps(s)
            s.set(outcome["status"], outcome["message"])
            return {"ok": True, "message": outcome["message"]}
        
        if cmd.name == "approve":
            return await self._submit(s, bool(cmd.args.get("confirm_incomplete")))

        return {"ok": False, "message": f"Unknown command {cmd.name!r}"}

    async def _advance_steps(self, s: Session, max_steps: int = MAX_STEPS) -> dict:
        """Walk a multi-page application: fix the current page, press Next, fill the new page, repeat, and stop
        at the page whose only way forward is Submit (never pressed here). Every stop says exactly why."""
        advanced = 0
        s.final_reached, s.blockers = False, []
        for _ in range(max_steps):
            page = self._current_page(s)
            await self._refresh(s)
            retyped = await retype_rejected_values(page) if s.required_empty else []
            if retyped:
                await self._refresh(s)
            if [label for label in s.required_empty if label not in retyped]:  # re-typed ones: let Next decide
                return self._stopped(s, advanced, "blocked")
            next_btn = await self._find_button(page, NEXT_RX)
            if next_btn is None:
                if await self._find_button(page, FINAL_SUBMIT_RX) is not None:
                    await self._reach_final(s)
                    return self._stopped(s, advanced, "final")
                return self._stopped(s, advanced, "no_next")

            before = await self._step_signature(page)
            if not await self._press(page, next_btn):
                return self._stopped(s, advanced, "no_next")
            if not await self._moved_on(s, before):
                # The site refused the page: values it never registered get one more, slower try.
                await self._refresh(s)
                moved = False
                if await retype_rejected_values(page) and (btn := await self._find_button(page, NEXT_RX)):
                    await self._press(page, btn)
                    moved = await self._moved_on(s, before)
                if not moved:
                    # Slow sites sometimes arrive after we gave up: look once more before calling it stalled.
                    await page.wait_for_timeout(4000)
                    moved = await self._step_signature(self._current_page(s)) != before
                if not moved:
                    await self._refresh(s)
                    return self._stopped(s, advanced, "stalled")
                await page.wait_for_timeout(1500)
            page = self._current_page(s)

            advanced += 1
            s.step += 1
            s.step_name = await self._step_name(page)
            log.info("SESSION %s | advanced to step %s %s| %s", s.id, s.step,
                     f"'{s.step_name}' " if s.step_name else "", page.url)
            await dismiss_overlays(page)  # a new page can bring a new cookie layer that eats every click
            previous_rows = list(s.rows)
            result = await s.refill()  # a page without fields (the site's Review page) just adds nothing
            s.apply_result(result)
            s.rows = previous_rows + s.rows
            s.summary["needs_review"] = sum(1 for row in s.rows if row.get("needs_review"))
        await self._refresh(s)
        return self._stopped(s, advanced, "limit")

    def _stopped(self, s: Session, advanced: int, why: str) -> dict:
        moved = f"Filled {advanced} more page(s). " if advanced else ""
        status = "ready_for_review"
        if why == "final":
            status = "ready_to_submit"
            message = (f"{moved}Reached the final page (step {s.step}). Check every page below and the form in the "
                       "browser, then press 'Approve & submit'.")
        elif why in {"blocked", "stalled"}:
            s.blockers = [f"{r['label'] or '(unlabelled)'}: {r.get('error') or ('required' if not r['value'] else 'invalid')}"
                          for r in s.live_rows if (r["required"] and not r["value"]) or r.get("invalid")]
            what = "; ".join(s.blockers[:6]) or "the site did not say which field (look for red text in the browser)"
            lead = "The Next button did not move the form forward." if why == "stalled" else \
                f"Stopped on step {s.step} before pressing Next."
            message = f"{moved}{lead} Fix these in the browser, then press 'Fill next steps': {what}"
        elif why == "no_next":
            message = (f"{moved}No Next/Continue or Submit button found on step {s.step}. Finish this page in the "
                       "browser, then press 'Fill next steps'.")
        else:
            message = f"{moved}Stopped at the safety limit of {MAX_STEPS} pages. Check the browser before continuing."
        log.info("SESSION %s | step walk stopped (%s) | %s", s.id, why, message)
        return {"advanced": advanced, "status": status, "message": message, "why": why,
                "stalled": why == "stalled", "blocked_required": s.required_empty}

    async def _reach_final(self, s: Session) -> None:
        s.final_reached = True
        page = s.page
        try:
            s.review_text = re.sub(r"\n\s*\n+", "\n", await page.locator("main, [role=main], form, body").first
                                   .inner_text(timeout=3000))[:12000]
        except Exception:
            s.review_text = ""
        if self.screenshot_fn:
            s.screenshot = await self.screenshot_fn(page, "final_review")

    @staticmethod
    def _current_page(s: Session):
        pages = [p for p in s.context.pages if not p.is_closed()] if s.context else []
        s.page = pages[-1] if pages else s.page
        return s.page

    @staticmethod
    async def _find_button(page, pattern: re.Pattern):
        """First visible, enabled control whose text matches. Real buttons win over links, so a progress-bar
        link called 'Review' is never mistaken for the form's own button."""
        for selector in (BUTTONS, "a"):
            for frame in list(page.frames):
                try:
                    els = frame.locator(selector)
                    for i in range(min(await els.count(), 300)):
                        el = els.nth(i)
                        try:
                            if not await el.is_visible() or not await el.is_enabled():
                                continue
                            label = ((await el.get_attribute("value")) or (await el.inner_text()) or "").strip() \
                                or (await el.get_attribute("aria-label") or "").strip()
                        except Exception:
                            continue
                        if label and len(label) < 60 and pattern.match(label):
                            return el
                except Exception:
                    continue
        return None

    @staticmethod
    async def _press(page, el) -> bool:
        for attempt in range(2):
            try:
                await el.click(timeout=5000)
                break
            except Exception:
                if attempt:
                    return False
                await dismiss_overlays(page)  # a late cookie/consent layer can sit on top of the button
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            pass
        await page.wait_for_timeout(2500)
        return True

    async def _moved_on(self, s: Session, before: tuple, timeout_s: int = 40) -> bool:
        """Did Next take us to another page? Sites save each page to their server first (a spinner for a few
        seconds), so keep looking while the page is busy; a page that is idle and unchanged was refused."""
        waited = 0
        while True:
            page = self._current_page(s)
            if await self._step_signature(page) != before:
                for _ in range(10):  # let the new page finish loading before it is read
                    if not await self._busy(page):
                        break
                    await page.wait_for_timeout(1000)
                await page.wait_for_timeout(800)
                return True
            if waited >= timeout_s or (waited >= 14 and not await self._busy(page)):
                return False
            await page.wait_for_timeout(1000)
            waited += 1

    @staticmethod
    async def _busy(page) -> bool:
        try:
            return await page.evaluate(BUSY_JS)
        except Exception:
            return True  # mid-navigation: the old document is gone, the new one isn't ready

    @staticmethod
    async def _step_name(page) -> str:
        try:
            return await page.evaluate(STEP_NAME_JS)
        except Exception:
            return ""

    @staticmethod
    async def _step_signature(page) -> tuple:
        """Which page of the form is this? Field labels and headings only: values and the red error text a
        rejected Next adds must not look like a new page."""
        snap = await snapshot(page)
        try:
            headings = await page.evaluate(
                "() => [...document.querySelectorAll('h1, h2, h3, legend')].filter(e => e.offsetParent)"
                ".map(e => e.innerText.trim()).join('|').slice(0, 2000)")
        except Exception:
            headings = ""
        fields = tuple((row["label"], row["type"]) for row in snap["rows"])
        return page.url, await SessionManager._step_name(page), fields, headings


    async def _refresh(self, s: Session) -> None:
        snap = await snapshot(s.page)
        s.live_rows, s.required_empty = snap["rows"], snap["required_empty"]
        try:
            s.title = await s.page.title()
        except Exception:
            pass

    async def _submit(self, s: Session, confirm_incomplete: bool) -> dict:
        await self._refresh(s)
        if s.required_empty and not confirm_incomplete:
            return {"ok": False, "needs_confirm": True,
                    "message": "Required fields are incomplete or invalid: " + ", ".join(s.required_empty[:8])}
        s.set("submitting", "Clicking Submit...")
        if not await click_by_label(s.page, SUBMIT_RX):
            s.set("ready_for_review", "")
            return {"ok": False, "message": "No Submit button found. On a multi-step form use 'Fill next steps'; "
                                            "otherwise submit it yourself in the browser."}
        await s.page.wait_for_timeout(4000)
        if self.screenshot_fn:
            s.screenshot = await self.screenshot_fn(s.page, "after_submit")
        try:
            body = (await s.page.locator("body").inner_text(timeout=3000))[:6000]
        except Exception:
            body = ""
        if SUCCESS_RX.search(body):
            s.set("submitted", "Submitted. A confirmation message was detected.")
        else:
            s.set("submitted_unverified", "Submit was clicked but no confirmation text was detected. "
                                          "Check the browser window.")
        s.finished = True
        return {"ok": True, "message": s.message}