"""Review page + API for application sessions.

Safety notes
  * Every mutating call needs the per-session token in the `X-Session-Token` header.
    A custom header forces a CORS preflight, so a random website cannot make your
    browser POST to http://127.0.0.1:8000/.../approve and submit an application.
  * The page renders all form values with textContent (never innerHTML): they come
    from untrusted websites and from the LLM.
"""
from __future__ import annotations

import hmac
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.sessions import SessionManager

ACTIONS = {"refresh", "refill", "next", "approve", "cancel", "answer"}

def make_router(manager: SessionManager) -> APIRouter:
    router = APIRouter()

    def _session(session_id: int, token: Optional[str]):
        s = manager.get(session_id)
        if s is None:
            raise HTTPException(404, "Unknown session")
        if not token or not hmac.compare_digest(token, s.token):
            raise HTTPException(403, "Bad or missing session token")
        return s

    @router.get("/review/{session_id}", response_class=HTMLResponse)
    async def review_page(session_id: int, t: str = ""):
        _session(session_id, t)
        return HTMLResponse(REVIEW_HTML)

    @router.get("/sessions/{session_id}")
    async def state(session_id: int, request: Request, t: str = ""):
        s = _session(session_id, request.headers.get("x-session-token") or t)
        return s.public()

    @router.post("/sessions/{session_id}/{action}")
    async def act(session_id: int, action: str, request: Request):
        s = _session(session_id, request.headers.get("x-session-token"))  # header only (CSRF-safe)
        if action not in ACTIONS:
            raise HTTPException(404, "Unknown action")
        try:
            body = await request.json()
        except Exception:
            body = {}
        result = await manager.send(s, action, confirm_incomplete=bool(body.get("confirm_incomplete")), answers=body.get("answers") or [])
        return JSONResponse(result)

    return router


REVIEW_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Review application</title>
<style>
:root{--bg:#fff;--fg:#1c1f23;--mut:#667085;--line:#e4e7ec;--card:#f9fafb;--warn:#fff4d6;--bad:#fde8e8;--ok:#e6f6ec;--btn:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#14171a;--fg:#e8eaed;--mut:#98a2b3;--line:#2b3138;--card:#1b1f24;--warn:#3a3014;--bad:#3b1c1c;--ok:#16301f;--btn:#1b1f24}}
*{box-sizing:border-box}body{font-family:system-ui,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:24px;max-width:1100px;margin:auto}
h1{font-size:20px;margin:0 0 4px}.url{color:var(--mut);font-size:13px;word-break:break-all}
.badge{display:inline-block;padding:3px 10px;border-radius:99px;font-size:12px;font-weight:600;background:var(--card);border:1px solid var(--line)}
.row{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0;align-items:center}
button{padding:9px 14px;border:1px solid var(--line);background:var(--btn);color:var(--fg);border-radius:8px;cursor:pointer;font-size:14px}
button:disabled{opacity:.45;cursor:not-allowed}button.go{background:#12805c;color:#fff;border-color:#12805c;font-weight:600}
button.stop{color:#b42318}
.banner{padding:12px 14px;border-radius:8px;margin:12px 0;font-size:14px}.warn{background:var(--warn)}.bad{background:var(--bad)}.ok{background:var(--ok)}
table{width:100%;border-collapse:collapse;font-size:14px;margin:8px 0 24px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
tr.review td{background:var(--warn)}tr.problem td{background:var(--bad)}td.val{word-break:break-word;max-width:380px}
.mut{color:var(--mut)}h2{font-size:15px;margin:20px 0 0}details summary{cursor:pointer;margin:8px 0}
</style></head><body>
<h1 id="title">Application</h1><div class="url" id="url"></div>
<div class="row"><span class="badge" id="status">starting</span><span class="mut" id="msg"></span></div>
<div id="banners">
<div id="missing"></div></div>
<div class="row">
  <button id="b-refresh">Refresh from browser</button>
  <button id="b-refill">Re-scan &amp; fill</button>
  <button id="b-next">Fill next steps</button>
  <button id="b-approve" class="go">Approve &amp; submit</button>
  <button id="b-cancel" class="stop">Cancel &amp; close browser</button>
</div>
<p class="mut" style="margin-top:-6px;font-size:13px">You can edit any field directly in the browser window, then press
<b>Refresh from browser</b>. Nothing is submitted until you press <b>Approve &amp; submit</b>.</p>
<h2>Filled by the automation</h2>
<table><thead><tr><th>Page</th><th>Field</th><th>Value</th><th>Source</th><th>Conf.</th><th>Result</th></tr></thead><tbody id="rows"></tbody></table>
<details id="final-wrap" hidden open><summary><b>The site's final review page</b> <span class="mut">(what it will submit)</span></summary>
<pre id="final-text" style="white-space:pre-wrap;font:13px/1.45 system-ui,Segoe UI,Arial,sans-serif;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;max-height:520px;overflow:auto"></pre></details>
<details id="live-wrap"><summary><b>Form as it is right now</b> <span class="mut">(after Refresh)</span></summary>
<table><thead><tr><th>Field</th><th>Current value</th><th>Required</th><th>Problem</th></tr></thead><tbody id="live"></tbody></table></details>
<script>
const sid = location.pathname.split('/').pop();
const T = new URLSearchParams(location.search).get('t') || '';
const $ = id => document.getElementById(id);
let last = null, refreshedOnce = false;

async function api(path, method, body) {
  const r = await fetch(path, {method: method || 'GET', headers: {'X-Session-Token': T, 'Content-Type': 'application/json'},
                               body: body ? JSON.stringify(body) : undefined});
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}
function el(tag, text, cls) { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; }
function banner(text, cls) { const d = el('div', text, 'banner ' + cls); $('banners').appendChild(d); }

function render(s) {
  last = s;
  $('title').textContent = s.title || 'Application';
  $('url').textContent = s.url;
  $('status').textContent = s.status.replace(/_/g, ' ');
  $('msg').textContent = s.message || '';
  $('banners').replaceChildren();
  if (s.finished && (s.status === 'submitted')) banner('Submitted. ' + (s.message || ''), 'ok');
  else if (s.status === 'submitted_unverified') banner(s.message, 'warn');
  else if (s.status === 'error') banner('Something went wrong: ' + s.message, 'bad');
  if (s.final_reached && !s.finished)
    banner('All pages are filled and the form is on its final page (step ' + s.step + '). Check every row below and the browser window, then press Approve & submit.', 'ok');
  if (s.blockers && s.blockers.length && !s.finished)
    banner('Blocking the next page: ' + s.blockers.join('; '), 'bad');
  else if (s.required_empty && s.required_empty.length && !s.finished)
    banner('Required fields incomplete or invalid: ' + s.required_empty.join(', '), 'bad');
  if (s.summary && s.summary.needs_review && !s.finished)
    banner(s.summary.needs_review + ' field(s) are highlighted: please check them before approving.', 'warn');

  const rows = $('rows'); rows.replaceChildren();
  (s.rows || []).forEach(r => {
    const tr = el('tr'); const bad = ['error', 'verified_failed'].includes(r.status);
    if (bad) tr.className = 'problem'; else if (r.needs_review) tr.className = 'review';
    tr.append(el('td', String(r.step || 1) + (r.step_name ? ' - ' + r.step_name : ''), 'mut'),
              el('td', (r.label || '(unlabelled)') + (r.required ? ' *' : '')), el('td', r.value == null ? '' : String(r.value), 'val'),
              el('td', r.source), el('td', r.confidence ? String(r.confidence) : ''),
              el('td', r.status + (r.detail ? ' - ' + r.detail : '')));
    rows.appendChild(tr);
  });
  const live = $('live'); live.replaceChildren();
  (s.live_rows || []).forEach(r => {
    const tr = el('tr'); if ((r.required && !r.value) || r.invalid) tr.className = 'problem';
    tr.append(el('td', r.label || '(unlabelled)'), el('td', r.value, 'val'), el('td', r.required ? 'yes' : ''),
              el('td', r.error || (r.invalid ? 'invalid' : '')));
    live.appendChild(tr);
  });
  $('final-wrap').hidden = !s.review_text;
  $('final-text').textContent = s.review_text || '';
  const can = s.ready && !s.busy;
  ['b-refresh', 'b-refill', 'b-next', 'b-approve'].forEach(id => $(id).disabled = !can);
  $('b-cancel').disabled = s.finished && !s.ready;
  $('b-cancel').textContent = s.finished ? 'Close browser' : 'Cancel & close browser';
  renderMissing(s);
}

let missingKey = '';
function renderMissing(s) {
  const key = JSON.stringify(s.missing || []);
  if (key === missingKey) return;
  missingKey = key;
  const box = $('missing'); box.replaceChildren();
  if (!(s.missing || []).length) return;
  box.appendChild(el('h2', 'I need these from you (saved to your profile for next time)'));
  s.missing.forEach(label => {
    const r = el('div', undefined, 'row');
    r.appendChild(el('label', label));
    const i = document.createElement('input');
    i.dataset.label = label; i.style.cssText = 'flex:1;min-width:220px;padding:8px';
    r.appendChild(i); box.appendChild(r);
  });
  const b = el('button', 'Save & continue', 'go');
  b.onclick = () => run('answer', {answers: [...box.querySelectorAll('input')]
    .filter(i => i.value.trim()).map(i => ({label: i.dataset.label, value: i.value}))});
  box.appendChild(b);
}

async function tick() {
  try {
    const s = await api('/sessions/' + sid);
    render(s);
    if (s.ready && !refreshedOnce) { refreshedOnce = true; run('refresh'); }
  } catch (e) { $('msg').textContent = 'Lost connection to the server (' + e.message + ')'; }
}
async function run(action, body) {
  try {
    const res = await api('/sessions/' + sid + '/' + action, 'POST', body);
    if (res.message && !res.ok) $('msg').textContent = res.message;
    await tick();
    return res;
  } catch (e) { $('msg').textContent = 'Action failed: ' + e.message; }
}
$('b-refresh').onclick = () => run('refresh');
$('b-refill').onclick = () => run('refill');
$('b-next').onclick = () => run('next');
$('b-cancel').onclick = () => { if (confirm('Close the browser and stop this application?')) run('cancel'); };
$('b-approve').onclick = async () => {
  if (!confirm('Submit this application now?')) return;
  let res = await run('approve');
  if (res && res.needs_confirm && confirm(res.message + '\n\nSubmit anyway?')) res = await run('approve', {confirm_incomplete: true});
};
tick(); setInterval(tick, 2000);
</script></body></html>
"""