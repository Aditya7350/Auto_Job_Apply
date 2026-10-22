"""Applies PlannedActions to the page and verifies each one.

Handles plain inputs, selects, radio groups, checkbox groups, file uploads and
custom comboboxes (React-select style: click, type, pick the option).
Nothing here ever clicks Submit.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.extractor import COMBO_VALUE_JS, Field, Option
from app.field_mapper import norm
from app.planner import FILL_ABOVE, PlannedAction, match_option

log = logging.getLogger("auto_job_apply")


@dataclass
class FillResult:
    handle: str
    label: str
    status: str  # filled | verified_failed | skipped | error
    detail: str = ""
    stale: bool = False  # the element vanished (site re-rendered it) -> caller may re-extract and retry


def _same_text(got: str, want: str) -> bool:
    """Did the field keep what we typed? Widgets reformat input: phone masks ("73502 68634", "+91 73502..."),
    postal codes ("411 001"), dates. Compare letters and digits only; fall back to a plain comparison for
    scripts the compact form would erase."""
    if (want or "").strip().casefold() in (got or "").casefold():
        return True
    g, w = (re.sub(r"[^a-z0-9]", "", (s or "").lower()) for s in (got, want))
    return bool(w) and w in g


async def _type_like_a_person(loc, value: str) -> None:
    await loc.click(timeout=3000)
    await loc.press("Control+A")
    await loc.press("Delete")
    await loc.press_sequentially(value, delay=15)


async def _fill_text(loc, value: str) -> bool:
    async def kept() -> bool:
        try:  # leaving the box is what makes most sites re-validate it (and drop a stale "Please enter..." error)
            await loc.blur(timeout=1000)
        except Exception:
            pass
        await loc.page.wait_for_timeout(400)  # React forms that lost the text re-render it away ~250 ms after blur
        got =await loc.input_value() if await loc.evaluate("e => 'value' in e") else await loc.inner_text()
        return _same_text(got, value)

    try:
        await loc.fill(value, timeout=4000)
    except Exception:  # some masked/custom inputs reject fill()
        await _type_like_a_person(loc, value)
        return await kept()
    if await kept():
        return True
    # The site threw the text away (its own state never saw a real keystroke): type it key by key.
    await _type_like_a_person(loc, value)
    return await kept()


async def _fill_combobox(page, frame, loc, value: str) -> bool:
    """Click, type to filter, wait for the (possibly async) suggestions, pick the best match, verify."""
    want = norm(value)
    await loc.click(timeout=3000)
    try:
        await loc.fill("")
    except Exception:
        pass
    await loc.type(value[:40], delay=40)
    pick = None
    for _ in range(10):  # type-ahead boxes (city search) fetch suggestions: allow ~4 seconds
        await page.wait_for_timeout(400)
        options = frame.locator('[role="option"]:visible')
        n = await options.count()
        texts = [(await options.nth(i).inner_text()).strip() for i in range(min(n, 30))]
        for test in (lambda t: norm(t) == want, lambda t: norm(t).startswith(want + " "),
                     lambda t: f" {want} " in f" {norm(t)} ", lambda t: want in norm(t)):
            idx = next((i for i, t in enumerate(texts) if test(t)), None)
            if idx is not None:
                pick = (options.nth(idx), texts[idx])
                break
        if pick:
            break
    if not pick:
        await loc.press("Escape")
        return False
    await pick[0].click(timeout=3000)
    await page.wait_for_timeout(250)
    try:  # what does the dropdown show now? (not every widget exposes it)
        shown = norm(await loc.evaluate(COMBO_VALUE_JS))
    except Exception:
        shown = ""
    return True if not shown else (norm(pick[1]) in shown or shown in norm(pick[1]) or want in shown)


async def _set_choice(f: Field, label: str) -> bool:
    opt = match_option(label, f.options)
    if not opt:
        return False
    loc = f.locator(opt)
    try:
        await loc.check(timeout=3000, force=True)
    except Exception:
        await loc.click(timeout=3000, force=True)  # custom-styled radios
    return await loc.is_checked()


async def _perform(page, f: Field, a: PlannedAction) -> bool:
    loc = f.locator()
    try:  # best effort: fill/click scroll by themselves, a stuck scroll must not block the fill
        await loc.scroll_into_view_if_needed(timeout=1500)
    except Exception:
        pass
    if a.action == "upload":
        await loc.set_input_files(a.value)
        return True
    if f.type == "select":
        try:
            await loc.select_option(label=a.value, timeout=3000)
        except Exception:
            opt = match_option(a.value, f.options)
            await loc.select_option(value=opt.value, timeout=3000)
        return True
    if f.type == "radio":
        return await _set_choice(f, a.value)
    if f.type == "checkbox_group":
        return all([await _set_choice(f, part.strip()) for part in a.value.split(";") if part.strip()])
    if f.type == "checkbox":
        await loc.set_checked(a.value == "true", timeout=3000, force=True)
        return await loc.is_checked() == (a.value == "true")
    if f.type == "combobox":
        return await _fill_combobox(page, f.frame, loc, a.value)
    return await _fill_text(loc, a.value)


async def apply_action(page, f: Field, a: PlannedAction) -> FillResult:
    if a.action == "skip" or a.value is None or (a.source == "llm" and a.confidence < FILL_ABOVE):
        return FillResult(f.handle, f.label, "skipped", a.reason)
    if await f.locator().count() == 0:  # the site replaced this element since we read the page
        return FillResult(f.handle, f.label, "error", "element is no longer on the page", stale=True)
    last = ""
    for attempt in (1, 2):  # pages that just loaded are still settling: one patient retry
        try:
            ok = await _perform(page, f, a)
            return FillResult(f.handle, f.label, "filled" if ok else "verified_failed",
                              "" if ok else "value did not stick; needs manual entry")
        except Exception as exc:
            stale = "data-aja-id" in str(exc)
            last = f"{type(exc).__name__}: {str(exc)[:160]}"
            if attempt == 1 and "Timeout" in type(exc).__name__:
                await page.wait_for_timeout(800)
                continue
            return FillResult(f.handle, f.label, "error", last, stale=stale)
    return FillResult(f.handle, f.label, "error", last)


def _occurrence_keys(fields: list[Field]) -> dict[tuple, Field]:
    """(label, type, n-th occurrence) -> field; survives handles being re-assigned."""
    seen: dict[tuple, int] = {}
    out = {}
    for fld in fields:
        base = (norm(fld.label), fld.type)
        seen[base] = seen.get(base, 0) + 1
        out[base + (seen[base],)] = fld
    return out


async def _remap(page, old_fields: list[Field]) -> dict[str, Field]:
    """Frameworks like React replace DOM nodes after an edit, dropping our tag. Re-extract and match
    each old field to its new element by label/type/occurrence."""
    from app.extractor import extract_fields
    new = _occurrence_keys(await extract_fields(page))
    mapping = {}
    for key, old in _occurrence_keys(old_fields).items():
        if key in new:
            nf = new[key]
            nf.options = nf.options or old.options  # keep options we already probed
            mapping[old.handle] = nf
    return mapping


async def apply_plan(page, fields: list[Field], actions: list[PlannedAction]) -> list[FillResult]:
    by_handle = {f.handle: f for f in fields}
    remapped: dict[str, Field] | None = None
    results = []
    for a in actions:
        f = by_handle.get(a.handle)
        if f is None:
            continue
        r = await apply_action(page, f, a)
        if r.status == "error" and r.stale:
            if remapped is None:
                remapped = await _remap(page, fields)
            nf = remapped.get(a.handle)
            if nf is not None:
                log.info("FILL | element was re-rendered by the site; retrying %s", a.label[:50])
                r = await apply_action(page, nf, a)
        log.info("FILL | %s | %s | %s | source=%s %s", r.status, a.label[:60], a.field_type, a.source, r.detail)
        results.append(r)
    return results