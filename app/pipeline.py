"""One call that does: read the form -> plan answers -> fill -> return review rows.

The returned rows are what the review page shows. Nothing is submitted here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.extractor import COMBO_VALUE_JS, extract_fields, probe_combobox_options
from app.filler import _occurrence_keys, apply_action, apply_plan
from app.planner import plan_fields


def _rows(plan, results, fields) -> list[dict]:
    required = {f.handle: f.required for f in fields}
    rows = []
    for a in plan:
        r = results.get(a.handle)
        status = "kept" if a.source == "page" else (r.status if r else "skipped")
        broken = status in {"error", "verified_failed"}
        # Highlight what needs the user: failures, AI-written answers, consent/legal items, and REQUIRED blanks.
        # An optional field that was simply left empty (a cover letter...) is not a reason to worry.
        attention = broken or (status == "skipped" and required.get(a.handle, False)) or \
            (status != "skipped" and a.needs_review)
        rows.append({
            "handle": a.handle, "label": a.label, "type": a.field_type, "required": required.get(a.handle, False),
            "value": (a.value if a.field_type != "file" else ("(resume)" if status == "filled" else "")),
            "source": a.source, "confidence": round(a.confidence, 2),
            "status": status, "detail": (r.detail if r and r.detail else a.reason),
            "needs_review": status != "kept" and attention,
        })
    return rows


async def fill_application(page, profile: dict, resume_path: Optional[Path] = None,
                           job: Optional[dict] = None, llm=None, max_passes: int = 3) -> dict:
    """Fill the application in the page's main content.

    Forms grow as you fill them ("If you chose Other, please specify", a dependent "Source" dropdown...), so
    after each pass that changed something we look again and handle only the fields that are NEW.
    """
    handled: set = set()  # (label, type, n-th occurrence) of every field already dealt with
    rows: list[dict] = []
    for pass_no in range(max_passes):
        fields = await extract_fields(page)
        # A page that just loaded (step 2 after Next) may not have rendered its fields yet: wait for them
        # instead of concluding the page is empty.
        for _ in range(16):
            if pass_no > 0 or len([f for f in fields if f.type != "password"]) >= 2:
                break
            await page.wait_for_timeout(500)
            fields = await extract_fields(page)
        # Dependent dropdowns (Mastercard "Source") appear first and load their choices a moment later.
        # Wait for them instead of planning against an empty list and never looking again.
        for _ in range(10):
            pending = [f for f in fields if f.type == "select" and not f.real_options()
                       and not (f.value or "").strip() and f.required]
            if not pending or pass_no == 0 and _ >= 2:
                break
            await page.wait_for_timeout(500)
            fields = await extract_fields(page)
        keyed = _occurrence_keys(fields)
        fresh = [f for key, f in keyed.items() if key not in handled]
        handled |= set(keyed)
        if not fresh:
            break
        await probe_combobox_options(fresh)  # learn each new dropdown's real choices
        plan = await plan_fields(fresh, profile, resume_path=resume_path, job=job, llm=llm)
        results = {r.handle: r for r in await apply_plan(page, fields, plan)}
        await _repair_wiped_values(page, fields, plan, results)
        new_rows = _rows(plan, results, fields)
        rows += new_rows if pass_no == 0 else [r for r in new_rows if r["status"] != "kept"]
        if not any(r.status == "filled" for r in results.values()):
            break  # nothing changed, so nothing new can have appeared
        await page.wait_for_timeout(500)  # let dependent fields render

    snap = await snapshot(page)  # the truth about the form right now, not a prediction
    return {
        "rows": rows,
        "filled": sum(1 for x in rows if x["status"] == "filled"),
        "needs_review": sum(1 for x in rows if x["needs_review"]),
        "required_unfilled": snap["required_empty"],
    }


async def retype_rejected_values(page) -> list[str]:
    """Text boxes that show a value the site still calls invalid ("Please enter email address" under a filled
    email): the site's own state never saw the text. Type it again key by key, like a person, then leave the
    box. Returns the labels that were re-typed: many sites only drop the error on the next Next/Submit, so
    pressing it again is the real check."""
    retyped = []
    for f in await extract_fields(page):
        if f.type not in TEXTLIKE_TYPES or not (f.value or "").strip():
            continue
        loc = f.locator()
        try:
            if not (await loc.evaluate(FIELD_ERROR_JS))["invalid"]:
                continue
            await loc.click(timeout=3000)
            await loc.press("Control+A")
            await loc.press("Delete")
            await loc.press_sequentially(f.value, delay=20)
            await loc.blur(timeout=1000)
            await page.wait_for_timeout(300)
            retyped.append(f.label)
        except Exception:
            continue
    return retyped


TEXTLIKE_TYPES = {"text", "email", "tel", "url", "number", "search", "textarea"}
_REPAIRABLE = {"text", "email", "tel", "url", "number", "textarea", "combobox", "select"}


async def _current_value(f) -> str:
    try:
        if f.type == "combobox":
            return (await f.locator().evaluate(COMBO_VALUE_JS)) or (await f.locator().input_value())
        return (await f.locator().input_value()) or ""
    except Exception:
        return ""


async def _repair_wiped_values(page, fields, plan, results) -> int:
    """Frameworks (React, Workday...) sometimes clear a field after LATER edits, so "filled" can be a lie by the
    time the form is done. Re-read the form, re-type whatever vanished once, and say so if the site keeps
    clearing it. Returns how many fields were re-filled."""
    await page.wait_for_timeout(700)
    fresh = _occurrence_keys(await extract_fields(page))
    old_key = {f.handle: key for key, f in _occurrence_keys(fields).items()}
    repaired = 0
    for a in plan:
        r = results.get(a.handle)
        if not r or r.status != "filled" or a.action not in {"fill", "select"} or a.field_type not in _REPAIRABLE:
            continue
        nf = fresh.get(old_key.get(a.handle))
        if nf is None or (nf.value or "").strip():
            continue  # still there (or the field is gone: not our business)
        nf.options = nf.options or next((f.options for f in fields if f.handle == a.handle), [])
        retry = await apply_action(page, nf, a)
        await page.wait_for_timeout(500)
        if retry.status == "filled" and (await _current_value(nf)).strip():
            r.detail, repaired = "re-filled: the site had cleared the value", repaired + 1
        else:
            r.status, r.detail = "verified_failed", "the site keeps clearing this value; type it in the browser"
    return repaired


# Is the field invalid, by the browser's rules OR the site's own (React forms keep their own state: the box can
# show text while the site says "Please enter email address")? Returns the site's message when it shows one.
FIELD_ERROR_JS = r"""
e => {
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  const shown = n => n && n.offsetParent !== null && clean(n.innerText);
  const native = !!e.willValidate && !e.checkValidity();
  const ids = [e.getAttribute('aria-errormessage'), e.getAttribute('aria-describedby')].join(' ').split(/\s+/);
  for (const id of ids) {
    const n = id && document.getElementById(id);
    if (shown(n) && /error|invalid|alert/i.test(n.id + ' ' + n.className + ' ' + (n.getAttribute('role') || '')))
      return { invalid: true, error: clean(n.innerText).slice(0, 160) };
  }
  let box = e.parentElement;  // the field's own row: stop before it grows to hold other fields
  for (let d = 0; d < 4 && box; d++, box = box.parentElement) {
    if (box.querySelectorAll('input:not([type=hidden]):not([type=radio]):not([type=checkbox]), select, textarea').length > 1) break;
    const msg = [...box.querySelectorAll('[class*="error" i], [id*="error" i], [role="alert"], .invalid-feedback')]
      .find(n => n !== e && !n.contains(e) && shown(n));
    if (msg) return { invalid: true, error: clean(msg.innerText).slice(0, 160) };
  }
  const aria = e.getAttribute('aria-invalid') === 'true';
  return { invalid: native || aria, error: native ? clean(e.validationMessage).slice(0, 160) : '' };
}
"""


async def snapshot(page) -> dict:
    """What the form looks like RIGHT NOW (including edits the user made by hand in the browser)."""
    fields = await extract_fields(page)
    rows, required_empty = [], []
    for f in fields:
        if f.type == "file":
            try:
                value = "(file attached)" if await f.locator().evaluate("e => e.files && e.files.length > 0") else ""
            except Exception:
                value = ""
        elif f.type == "checkbox":
            value = "checked" if f.value == "true" else ""
        else:
            value = (f.value or "").strip()
        try:
            check = await f.locator().evaluate(FIELD_ERROR_JS)
        except Exception:
            check = {"invalid": False, "error": ""}
        rows.append({"label": f.label, "type": f.type, "value": value, "required": f.required,
                     "invalid": check["invalid"], "error": check["error"]})
        if (f.required and not value) or check["invalid"]:
            required_empty.append(f.label or f.name or f.type)
    return {"rows": rows, "required_empty": required_empty}