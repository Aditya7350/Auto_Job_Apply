"""End-to-end test of the session/review flow against a local fake job site.

No internet, no real LLM (no API key is set, so unresolved fields stay for manual input).
Run:  pytest -q tests/test_integration.py
"""
import asyncio
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ["HEADLESS"] = "1"  # must be set before app.main is imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402
import pytest  # noqa: E402

JOB = "<h1>AI Engineer</h1><p>Build LLM apps with us.</p><a href='/form'>Apply Now</a>"
FORM = """<h1>Application</h1>
<form id=f>
 <label for=fn>First name *</label><input id=fn required>
 <label for=ln>Last name</label><input id=ln>
 <label for=em>Email</label><input id=em type=email>
 <label for=ph>Phone</label><input id=ph type=tel>
 <label for=co>Country</label><select id=co><option value=''>Select</option><option>India</option><option>United States</option></select>
 <label for=why>Why do you want this role? *</label><textarea id=why required></textarea>
 <button type=submit>Submit application</button>
</form>
<script>document.getElementById('f').addEventListener('submit', async e => {
  e.preventDefault(); await fetch('/submit', {method:'POST'});
  document.body.textContent = 'Thank you! Your application has been submitted.'; });</script>"""


class Site:
    def __init__(self):
        outer = self
        self.submissions = 0

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = {"/job": JOB, "/form": FORM}.get(urlparse(self.path).path, "<p>404</p>")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body.encode())

            def do_POST(self):
                outer.submissions += 1
                self.send_response(200)
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


async def wait_for(cond, timeout=60, step=0.3):
    waited = 0.0
    while waited < timeout:
        if cond():
            return True
        await asyncio.sleep(step)
        waited += step
    return False


def test_full_review_flow(monkeypatch):
    from tests.test_engine import PROFILE
    import app.main as m

    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    tmp = Path(tempfile.mkdtemp())
    resume = tmp / "resume.pdf"
    resume.write_bytes(b"%PDF-1.4 test")
    monkeypatch.setattr(m, "load_profile", lambda: PROFILE)
    monkeypatch.setattr(m, "resolve_resume_path", lambda p: resume)
    monkeypatch.setattr(m, "DB", tmp / "t.db")
    monkeypatch.setattr(m, "LOG_DIR", tmp)
    site = Site()

    async def go():
        try:
            await body()
        finally:
            await m.manager.shutdown()  # graceful, so a failed assertion cannot hang the test run

    async def body():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=m.app), base_url="http://t") as c:
            r = await c.post("/apply", data={"url": site.url + "/job"}, follow_redirects=False)
            assert r.status_code == 303
            loc = urlparse(r.headers["location"])
            sid, token = int(loc.path.split("/")[2]), parse_qs(loc.query)["t"][0]
            hdr = {"X-Session-Token": token}
            s = m.manager.get(sid)

            assert await wait_for(lambda: s.status in {"ready_for_review", "error", "apply_button_not_found"}), s.status
            assert s.status == "ready_for_review", (s.status, s.message)

            st = (await c.get(f"/sessions/{sid}", headers=hdr)).json()
            by = {r["label"]: r for r in st["rows"]}
            assert by["First name"]["status"] == "filled" and by["First name"]["source"] == "profile"
            assert by["Country"]["status"] == "filled"
            assert by["Why do you want this role?"]["needs_review"], "unanswered question must be flagged"
            assert await s.page.input_value("#fn") == "Asha"

            # --- security: no token / wrong token can never act
            assert (await c.post(f"/sessions/{sid}/approve")).status_code == 403
            assert (await c.post(f"/sessions/{sid}/approve", headers={"X-Session-Token": "nope"})).status_code == 403
            assert (await c.get(f"/review/{sid}")).status_code == 403
            assert (await c.get(f"/sessions/{sid}")).status_code == 403
            page_html = await c.get(f"/review/{sid}?t={token}")
            assert page_html.status_code == 200 and "textContent" in page_html.text
            assert site.submissions == 0

            # --- incomplete form: approve is refused until the user confirms
            await c.post(f"/sessions/{sid}/refresh", headers=hdr)
            res = (await c.post(f"/sessions/{sid}/approve", headers=hdr)).json()
            assert res["ok"] is False and res["needs_confirm"] and "Why do you want" in res["message"]
            assert site.submissions == 0

            # --- the user edits in the browser; refresh sees it; re-scan never overwrites it
            await s.page.fill("#why", "I enjoy building LLM products.")
            await s.page.fill("#fn", "Custom")
            await c.post(f"/sessions/{sid}/refresh", headers=hdr)
            st = (await c.get(f"/sessions/{sid}", headers=hdr)).json()
            assert st["required_empty"] == []
            await c.post(f"/sessions/{sid}/refill", headers=hdr)
            assert await s.page.input_value("#fn") == "Custom", "re-scan must not overwrite user edits"
            assert await s.page.input_value("#why") == "I enjoy building LLM products."
            st = (await c.get(f"/sessions/{sid}", headers=hdr)).json()
            by = {r["label"]: r for r in st["rows"]}
            assert by["Email"]["status"] == "filled" and by["Email"]["source"] == "profile", "re-scan keeps provenance"
            assert st["summary"]["filled"] >= 5, "re-scan must not reset the filled count"

            # --- approve: submits exactly once and detects the confirmation
            res = (await c.post(f"/sessions/{sid}/approve", headers=hdr)).json()
            assert res["ok"], res
            assert s.status == "submitted" and site.submissions == 1
            again = (await c.post(f"/sessions/{sid}/approve", headers=hdr)).json()
            assert again["ok"] is False and site.submissions == 1, "a finished application cannot be re-submitted"

            row = __import__("sqlite3").connect(tmp / "t.db").execute(
                "SELECT status, fields_filled FROM applications WHERE id=?", (sid,)).fetchone()
            assert row[0] == "submitted" and row[1] >= 4

            await c.post(f"/sessions/{sid}/cancel", headers=hdr)  # "Close browser"
            assert await wait_for(lambda: s.task.done(), 15)

    try:
        asyncio.run(go())
    finally:
        site.close()


def test_cancel_closes_browser_without_submitting(monkeypatch):
    from tests.test_engine import PROFILE
    import app.main as m

    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    tmp = Path(tempfile.mkdtemp())
    monkeypatch.setattr(m, "load_profile", lambda: PROFILE)
    monkeypatch.setattr(m, "resolve_resume_path", lambda p: None)
    monkeypatch.setattr(m, "DB", tmp / "t.db")
    monkeypatch.setattr(m, "LOG_DIR", tmp)
    site = Site()

    async def go():
        try:
            await body()
        finally:
            await m.manager.shutdown()

    async def body():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=m.app), base_url="http://t") as c:
            r = await c.post("/apply", data={"url": site.url + "/job"}, follow_redirects=False)
            loc = urlparse(r.headers["location"])
            sid, token = int(loc.path.split("/")[2]), parse_qs(loc.query)["t"][0]
            s = m.manager.get(sid)
            assert await wait_for(lambda: s.status == "ready_for_review")
            res = (await c.post(f"/sessions/{sid}/cancel", headers={"X-Session-Token": token})).json()
            assert res["ok"]
            assert await wait_for(lambda: s.task.done(), 15)
            assert s.status == "cancelled" and site.submissions == 0
            assert s.browser.is_connected() is False

    try:
        asyncio.run(go())
    finally:
        site.close()


def test_review_page_renders_untrusted_values_as_text_and_sends_token():
    """The page must never execute HTML that came from a website or from the LLM."""
    from playwright.async_api import async_playwright
    from app.review import REVIEW_HTML

    evil = "<img src=x onerror=\"window.__pwned=true\">"
    state = {
        "id": 7, "url": "https://jobs.example.com/apply", "title": "AI Engineer", "status": "ready_for_review",
        "message": "5 fields filled", "busy": False, "ready": True, "finished": False,
        "summary": {"filled": 5, "needs_review": 1}, "required_empty": ["Why do you want this role?"],
        "rows": [
            {"label": "First name", "type": "text", "value": "Asha", "source": "profile", "confidence": 0.95,
             "status": "filled", "detail": "", "needs_review": False},
            {"label": evil, "type": "textarea", "value": evil, "source": "llm", "confidence": 0.6,
             "status": "filled", "detail": evil, "needs_review": True},
        ],
        "live_rows": [{"label": "First name", "type": "text", "value": "Asha", "required": True}],
    }
    calls = []

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            page.on("dialog", lambda d: asyncio.ensure_future(d.accept()))

            async def route(r):
                req = r.request
                path = urlparse(req.url).path
                if path == "/review/7":
                    await r.fulfill(status=200, content_type="text/html", body=REVIEW_HTML)
                elif path == "/sessions/7" and req.method == "GET":
                    await r.fulfill(status=200, content_type="application/json", body=__import__("json").dumps(state))
                else:
                    calls.append((req.method, path, req.headers.get("x-session-token")))
                    await r.fulfill(status=200, content_type="application/json",
                                    body='{"ok": false, "needs_confirm": true, "message": "Required fields are incomplete or invalid: Why"}'
                                    if path.endswith("approve") and len(
                                        [c for c in calls if c[1].endswith("approve")]) == 1 else '{"ok": true}')

            await page.route("http://app.test/**", route)
            await page.goto("http://app.test/review/7?t=SECRET")
            await page.wait_for_selector("#rows tr")

            assert await page.evaluate("window.__pwned === undefined"), "HTML from the form was executed!"
            assert await page.locator("#rows img").count() == 0
            assert evil in await page.locator("#rows").inner_text(), "hostile value must be shown as plain text"
            assert "Required fields incomplete or invalid" in await page.locator("#banners").inner_text()
            assert await page.locator("#rows tr.review").count() == 1
            assert await page.locator("#b-approve").is_enabled()

            await page.click("#b-approve")  # first approve is refused -> page asks "submit anyway?" -> accepted
            await page.wait_for_timeout(600)
            approves = [c for c in calls if c[1].endswith("approve")]
            assert len(approves) == 2, calls
            assert all(c[2] == "SECRET" for c in calls), "every call must carry the session token header"
            await page.screenshot(path=str(Path(tempfile.gettempdir()) / "review_page.png"), full_page=True)
            await browser.close()

    asyncio.run(go())