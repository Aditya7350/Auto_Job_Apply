"""Structured form reader.

Turns whatever form is on the page (main document + child frames) into a flat
list of `Field` objects: label, type, required, options, current value and a
stable handle. The LLM only ever sees this description, never raw HTML, and
all actions are executed through the handle, so the model can't pick arbitrary
selectors.

Every control gets a `data-aja-id` attribute so it can be re-located later,
even when the site uses custom widgets, frames or shadow DOM (open roots).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# Runs inside each frame. Returns a list of plain dicts.
EXTRACT_JS = r"""
() => {
  document.querySelectorAll('[data-aja-id]').forEach(e => e.removeAttribute('data-aja-id'));
  let counter = 0;
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  const isVisible = el => {
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none') return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  };
  const tag = el => {
    if (!el.hasAttribute('data-aja-id')) el.setAttribute('data-aja-id', String(counter++));
    return el.getAttribute('data-aja-id');
  };
  const textOfIds = ids => (ids || '').split(/\s+/).map(id => {
    const n = id ? document.getElementById(id) : null;
    return n ? clean(n.innerText || n.textContent) : '';
  }).filter(Boolean).join(' ');

  const nearbyText = el => {
    let node = el;
    for (let d = 0; d < 4 && node && node.parentElement; d++) {
      node = node.parentElement;
      let best = null;
      for (const c of node.querySelectorAll('label, legend, [class*="label" i], [class*="question" i], h1, h2, h3, h4, h5, h6')) {
        if (c.contains(el)) continue;
        if (c.tagName === 'LABEL' && c.control && c.control !== el) continue;       // it labels a different field
        if (!(c.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING)) continue;  // must come before el
        best = c;                                                                    // the last one is the nearest
      }
      if (best) {
        const t = clean(best.innerText);
        if (t) return t.slice(0, 200);
      }
    }
    return '';
  };
  // A fieldset only names a choice group when it holds that group alone: many sites wrap the WHOLE form
  // in one <fieldset><legend>You are applying for...</legend>, which must not become every radio's question.
  const tightFieldset = fs => {
    const groups = new Set();  // one key per question: a radio/checkbox name, or the control itself
    for (const c of fs.querySelectorAll('input, select, textarea')) {
      if (['hidden', 'submit', 'button'].includes((c.type || '').toLowerCase())) continue;
      groups.add((c.type === 'radio' || c.type === 'checkbox') && c.name ? 'n:' + c.name : c);
    }
    return groups.size <= 1;
  };
  const groupQuestion = el => {
    for (let grp = el.closest('[role="radiogroup"], [role="group"]'); grp;
         grp = grp.parentElement && grp.parentElement.closest('[role="radiogroup"], [role="group"]')) {
      const l = textOfIds(grp.getAttribute('aria-labelledby')) || clean(grp.getAttribute('aria-label'));
      if (l) return l.slice(0, 250);
    }
    const fs = el.closest('fieldset');
    if (fs && tightFieldset(fs)) {
      const lg = fs.querySelector('legend'); if (lg && clean(lg.innerText)) return clean(lg.innerText).slice(0, 250);
    }
    return nearbyText(el);
  };
  const groupRequired = el => !!el.closest(
    '[role="radiogroup"][aria-required="true"], [role="group"][aria-required="true"], fieldset[aria-required="true"]');
  const ownLabel = el => {
    const lb = textOfIds(el.getAttribute('aria-labelledby'));
    if (lb) return lb;
    if (el.labels && el.labels.length) {
      const t = [...el.labels].map(l => clean(l.innerText)).filter(Boolean).join(' ');
      if (t) return t.slice(0, 250);
    }
    const al = clean(el.getAttribute('aria-label'));
    if (al) return al;
    return '';
  };

  const comboText = el => {
    let n = el;
    for (let d = 0; d < 5; d++) {
      n = n.parentElement;
      if (!n) break;
      if (n.querySelectorAll('[role="combobox"], select').length > 1) break;  // reached another field
      const single = n.querySelector('[class*="single-value" i], [class*="singleValue" i]');
      if (single) return clean(single.innerText);
      const multi = [...n.querySelectorAll('[class*="multi-value__label" i], [class*="multiValue" i] > div')];
      if (multi.length) return multi.map(x => clean(x.innerText)).join(', ');
    }
    return '';
  };
  const genericFile = /^(attach|upload|browse|choose|select file|drop|add file)/i;

  const NOISE_LABEL = /\bsave (this )?job\b|add to (saved|favou?rites?|wishlist)|\bremember me\b|keep me (signed|logged) in|^\s*select all\s*$|\bcompare jobs?\b/i;
  const attrsOf = n => ((n.getAttribute('class') || '') + ' ' + (n.id || '') + ' ' + (n.getAttribute('action') || '') + ' ' +
                        (n.getAttribute('aria-label') || '') + ' ' + (n.getAttribute('data-testid') || '')).toLowerCase();
  // Where on the page does this control live? Only "main" controls are part of an application.
  const regionOf = (el, label) => {
    if ((el.getAttribute('type') || '').toLowerCase() === 'search' || el.closest('[role="search"], search')) return 'search';
    const form = el.closest('form');
    if (form && /search/.test(attrsOf(form))) return 'search';
    const hint = ((el.getAttribute('placeholder') || '') + ' ' + (el.getAttribute('aria-label') || '') + ' ' +
                  (el.name || '') + ' ' + (el.id || '') + ' ' + (label || '')).trim();
    if (/^(search|keyword)|search (jobs?|job title|by|for|keyword)|job title or keyword/i.test(hint)) return 'search';
    for (let n = el; n && n !== document.body && n !== document.documentElement; n = n.parentElement) {
      const tn = n.tagName.toLowerCase(), role = n.getAttribute('role');
      if (tn === 'header' || tn === 'nav' || tn === 'footer' || role === 'banner' || role === 'navigation' ||
          role === 'contentinfo') return 'chrome';
      const a = attrsOf(n);
      if (/newsletter|subscribe|job-?alerts?|talent-?(community|network|pool)|mailing|email-?signup|signup-?email/.test(a)) return 'marketing';
      if (/chat|intercom|drift|livechat|zendesk|helpshift|webchat|messenger|assistant-widget/.test(a)) return 'chat';
      if (/onetrust|ot-sdk|cookie|cmp-|truste|didomi|usercentrics|cookiebot/.test(a)) return 'consent';
    }
    if (NOISE_LABEL.test(label || '')) return 'noise';
    return 'main';
  };

  const nodes = document.querySelectorAll(
    'input, textarea, select, [role="combobox"], [role="textbox"], [contenteditable="true"]');
  const fields = [];
  const radioGroups = new Map();
  const checkGroups = new Map();

  for (const el of nodes) {
    const tagName = el.tagName.toLowerCase();
    let type = tagName === 'input' ? (el.getAttribute('type') || 'text').toLowerCase() : tagName;
    if (el.getAttribute('role') === 'combobox' && tagName !== 'select') type = 'combobox';
    if (el.getAttribute('contenteditable') === 'true' || el.getAttribute('role') === 'textbox') {
      if (tagName !== 'input' && tagName !== 'textarea') type = 'textarea';
    }
    if (['hidden', 'submit', 'button', 'reset', 'image'].includes(type)) continue;
    if (el.disabled) continue;

    const choice = type === 'radio' || type === 'checkbox';
    // react-select & co. add an invisible, aria-hidden "required" twin next to the real input.
    if (!choice && type !== 'file') {
      if (el.closest('[aria-hidden="true"]') !== null) continue;
      if (parseFloat(getComputedStyle(el).opacity) === 0) continue;
    }
    const visible = isVisible(el) ||
      (choice && el.labels && [...el.labels].some(isVisible)) ||
      type === 'file';
    if (!visible) continue;

    const handle = tag(el);
    const lbl = ownLabel(el);

    if (type === 'radio') {
      const key = (el.form ? 'f' : '') + (el.name || ('solo-' + handle));
      let g = radioGroups.get(key);
      if (!g) {
        g = { handle, type: 'radio', label: groupQuestion(el), options: [], value: '',
              required: false, region: regionOf(el, ''), hint: '', name: el.name || '', placeholder: '', maxlength: 0 };
        radioGroups.set(key, g); fields.push(g);
      }
      const optLabel = lbl || clean(el.value) || clean(el.nextSibling && el.nextSibling.textContent);
      g.options.push({ label: optLabel, value: el.value, handle });
      if (el.checked) g.value = optLabel;
      if (el.required || el.getAttribute('aria-required') === 'true' || groupRequired(el)) g.required = true;
      continue;
    }

    if (type === 'checkbox') {
      const key = el.name ? ((el.form ? 'f' : '') + el.name) : '';
      const optLabel = lbl || clean(el.value);
      if (key && document.querySelectorAll('input[type="checkbox"][name="' + CSS.escape(el.name) + '"]').length > 1) {
        let g = checkGroups.get(key);
        if (!g) {
          g = { handle, type: 'checkbox_group', label: groupQuestion(el), options: [], value: '',
                required: false, region: regionOf(el, ''), hint: '', name: el.name, placeholder: '', maxlength: 0 };
          checkGroups.set(key, g); fields.push(g);
        }
        g.options.push({ label: optLabel, value: el.value, handle });
        if (el.required || groupRequired(el)) g.required = true;
        continue;
      }
      fields.push({ handle, type: 'checkbox', label: optLabel || groupQuestion(el), options: [],
        value: el.checked ? 'true' : 'false', required: el.required || el.getAttribute('aria-required') === 'true',
        region: regionOf(el, optLabel || groupQuestion(el)), hint: '', name: el.name || '', placeholder: '', maxlength: 0 });
      continue;
    }

    let label = lbl || clean(el.getAttribute('placeholder')) || nearbyText(el);
    if (type === 'file' && !label.trim()) label = 'File upload (' + (el.getAttribute('data-automation-id') || el.id || el.name || 'file') + ')';
    let options = [];
    if (tagName === 'select') {
      options = [...el.options].slice(0, 300).map(o => ({ label: clean(o.text), value: o.value })).filter(o => o.label);
    }
    const required = el.required || el.getAttribute('aria-required') === 'true' || /\*/.test(label);
    if (type === 'file' && genericFile.test(label.trim()) && (el.id || el.name)) {
      label = label.trim() + ' (' + (el.id || el.name) + ')';  // "Attach" twice is unreadable
    }
    let value = type === 'file' ? '' : (el.value !== undefined ? String(el.value) : clean(el.innerText));
    if (type === 'combobox' && !value) value = comboText(el);
    fields.push({
      handle, type, label: label.replace(/\s*\*\s*$/, ''), options, value,
      required, region: regionOf(el, label), hint: el.getAttribute('autocomplete') || '', name: el.name || el.id || '',
      placeholder: clean(el.getAttribute('placeholder')), maxlength: el.maxLength > 0 ? el.maxLength : 0
    });
  }
  for (const f of fields) {  // a trailing * means required, for every kind of control
    if (/\*\s*$/.test(f.label || '')) { f.required = true; f.label = f.label.replace(/\s*\*\s*$/, ''); }
  }
  return fields;
}
"""


_PLACEHOLDER_OPTION = re.compile(r"^\s*(select|choose|please (select|choose)|pick|--+|\u2014|\.\.\.)", re.I)


@dataclass
class Option:
    label: str
    value: str = ""
    handle: str = ""  # only set for radio / checkbox_group options


@dataclass
class Field:
    handle: str  # "<frame_index>:<n>"; unique within one extraction
    type: str  # text/email/tel/url/number/date/textarea/select/radio/checkbox/checkbox_group/combobox/file/password
    label: str
    required: bool = False
    options: list[Option] = field(default_factory=list)
    value: str = ""
    hint: str = ""  # autocomplete attribute
    name: str = ""
    placeholder: str = ""
    maxlength: int = 0
    region: str = "main"  # main | search | chrome | marketing | chat | consent | noise
    frame: Any = None  # Playwright Frame, not serialised

    def real_options(self) -> list:
        """Options without placeholders like "Select..." / "Choose one" (those are not answers)."""
        return [o for o in self.options if not _PLACEHOLDER_OPTION.match(o.label or "")]

    def to_prompt_dict(self) -> dict:
        d: dict = {"id": self.handle, "type": self.type, "label": self.label, "required": self.required}
        if self.real_options():
            d["options"] = [o.label for o in self.real_options()][:60]
        if self.hint:
            d["autocomplete"] = self.hint
        if self.maxlength:
            d["maxlength"] = self.maxlength
        if self.value and self.type != "file":
            d["current_value"] = self.value
        return d

    def locator(self, option: Optional[Option] = None):
        """Locator for this field (or one of its radio/checkbox options)."""
        handle = (option.handle if option and option.handle else self.handle)
        n = handle.split(":", 1)[1]
        return self.frame.locator(f'[data-aja-id="{n}"]').first


async def extract_fields(page, include_passwords: bool = False, regions: Optional[tuple] = ("main",)) -> list[Field]:
    """Extract visible, enabled form controls across all frames.

    By default only controls in the page's main content are returned: search bars, header/footer,
    newsletter and chat boxes, cookie dialogs and "save job" checkboxes are not part of an application.
    Pass regions=None to get everything (the login code needs that).
    """
    out: list[Field] = []
    for frame_index, frame in enumerate(list(page.frames)):
        try:
            if frame.is_detached():
                continue
            raw = await frame.evaluate(EXTRACT_JS)
        except Exception:
            continue
        for r in raw:
            if r["type"] == "password" and not include_passwords:
                continue
            if regions is not None and r.get("region", "main") not in regions:
                continue
            opts = [
                Option(label=o.get("label", ""), value=o.get("value", ""),
                       handle=f"{frame_index}:{o['handle']}" if o.get("handle") else "")
                for o in r.get("options", [])
            ]
            out.append(Field(
                handle=f"{frame_index}:{r['handle']}", type=r["type"], label=r.get("label", ""),
                required=bool(r.get("required")), options=opts, value=r.get("value", "") or "",
                hint=r.get("hint", ""), name=r.get("name", ""), placeholder=r.get("placeholder", ""),
                maxlength=int(r.get("maxlength") or 0), region=r.get("region", "main"), frame=frame,
            ))
    return out


_IDENTITY_KEYS = ("first_name", "last_name", "full_name", "email", "phone")


def identity_hits(fields: list[Field]) -> set[str]:
    """Which personal-detail fields (name, email, phone...) does this field list contain?"""
    from app.field_mapper import build_definitions, choose_field
    defs = [d for d in build_definitions({k: "x" for k in _IDENTITY_KEYS}) if d.key in _IDENTITY_KEYS]
    hits = set()
    for f in fields:
        if f.type in {"text", "email", "tel", "combobox"}:
            for text in (f.label, f.placeholder, f"{f.name} {f.hint}"):
                d = choose_field(text, defs) if text.strip() else None
                if d:
                    hits.add(d.key)
                    break
    return hits


def is_application_form(fields: list[Field]) -> bool:
    """A real application asks who you are AND takes a resume or many answers.

    Not enough on their own: a search box, a newsletter email, a login form, "save job" checkboxes.
    """
    answerable = [f for f in fields if f.type not in {"password"}]
    ident = identity_hits(answerable)
    has_file = any(f.type == "file" for f in answerable)
    return (has_file and len(ident) >= 1 and len(answerable) >= 3) or (len(ident) >= 2 and len(answerable) >= 4)


async def looks_like_application_form(page) -> bool:
    """True when the main content of the page holds a real application form."""
    return is_application_form(await extract_fields(page))


COMBO_VALUE_JS = """
el => {
  const clean = s => (s || '').replace(/\\s+/g, ' ').trim();
  let n = el;
  for (let d = 0; d < 5; d++) {
    n = n.parentElement;
    if (!n) break;
    if (n.querySelectorAll('[role="combobox"], select').length > 1) break;
    const single = n.querySelector('[class*="single-value" i], [class*="singleValue" i]');
    if (single) return clean(single.innerText);
    const multi = [...n.querySelectorAll('[class*="multi-value__label" i], [class*="multiValue" i] > div')];
    if (multi.length) return multi.map(x => clean(x.innerText)).join(', ');
  }
  return '';
}
"""

_READ_OPTIONS_JS = """
async (el) => {
  await new Promise(r => setTimeout(r, 300));
  const id = el.getAttribute('aria-controls') || el.getAttribute('aria-owns');
  const root = (id && document.getElementById(id)) || document;
  const seen = new Set(), out = [];
  for (const o of root.querySelectorAll('[role="option"]')) {
    const r = o.getBoundingClientRect();
    const t = (o.innerText || '').replace(/\\s+/g, ' ').trim();
    if (r.width > 0 && r.height > 0 && t && !seen.has(t)) { seen.add(t); out.push(t); }
    if (out.length >= 300) break;
  }
  return out;
}
"""


async def probe_combobox_options(fields: list[Field], max_probes: int = 40) -> int:
    """Dropdowns hide their choices until opened. Open each one, read the options, close it.

    Without this the LLM has to guess a value that may not exist in the list (it did: 'Company career
    page'). Async/type-ahead boxes (city search) show nothing until you type; they stay option-less.
    """
    probed = 0
    for f in fields:
        if f.type != "combobox" or f.options or f.value or probed >= max_probes:
            continue
        loc = f.locator()
        try:
            await loc.scroll_into_view_if_needed(timeout=2000)
            await loc.click(timeout=2000)
            texts = await loc.evaluate(_READ_OPTIONS_JS)
            f.options = [Option(label=t) for t in texts]
            probed += 1
        except Exception:
            pass
        finally:
            try:
                await loc.press("Escape", timeout=1000)
            except Exception:
                pass
    return probed