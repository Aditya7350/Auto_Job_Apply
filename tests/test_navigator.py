"""Universal navigation tests.

The synthetic site mirrors a real failure (a Mastercard careers page): a cookie dialog that blocks the page,
a chat panel covering the Apply button, a header search bar, a newsletter email box and "Save job"
checkboxes. The old code filled the search box and the newsletter box and never clicked Apply.
"""
import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from playwright.async_api import async_playwright  # noqa: E402

from app.extractor import extract_fields, looks_like_application_form  # noqa: E402
from app.navigator import apply_score, Clickable, decide_overlay_action, dismiss_overlays, navigate_to_application  # noqa: E402
from app.pipeline import fill_application  # noqa: E402
from tests.test_engine import PROFILE  # noqa: E402

HEADER = """<header><form class="search-bar"><input aria-label="Search job title" value="AI Engineer">
 <input aria-label="Location" value="Pune, Maharashtra, India"><button type="button">Search</button></form>
 <a href="#">Log in</a></header>"""
FOOTER = """<footer><form class="newsletter"><label>Enter Email address (Required)<input type="email"></label>
 <button type="button">Subscribe</button></form></footer>"""
SAVE = """<label><input type="checkbox"> Save job</label><label><input type="checkbox"> Save job Lead Software Engineer</label>"""
CHAT = """<div class="chat-widget" style="position:fixed;right:10px;top:10px;width:430px;height:520px;background:#ddd;z-index:50">
 <div>Career Assistant <button aria-label="Close chat" onclick="this.closest('.chat-widget').remove()">&times;</button></div>
 <p>Hi There! Are you interested in this job?</p></div>"""
COOKIES = """<div id="ot-banner" role="dialog" aria-modal="true" style="position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:100">
 <div style="background:#fff;margin:160px auto;width:620px;padding:24px"><h3>How we use cookies and your consent</h3>
 <p>We use cookies and similar technologies to improve the site.</p>
 <button onclick="document.title='manage'">Manage cookies</button>
 <button onclick="document.title='rejected';document.getElementById('ot-banner').remove()">Reject all</button>
 <button onclick="document.title='accepted';document.getElementById('ot-banner').remove()">Accept cookies</button></div></div>"""
APPLY_NEWTAB = '<a class="apply-btn" href="/form" target="_blank" style="position:absolute;right:80px;top:70px">Apply</a>'

FORM = """<h1>Application</h1>
<form id="f"><label for="fn">First Name*</label><input id="fn" required>
 <label for="ln">Last Name*</label><input id="ln" required>
 <label for="em">Email*</label><input id="em" type="email" required>
 <label for="ph">Phone</label><input id="ph" type="tel">
 <label for="cv">Resume/CV</label><input id="cv" type="file">
 <button type="submit">Submit application</button></form>"""


def page(body, cookies=False, chat=False, extra=""):
    return f"<html><head><title>Job</title></head><body>{HEADER}{body}{FOOTER}{COOKIES if cookies else ''}{CHAT if chat else ''}{extra}</body></html>"


PAGES = {
    "/job": page("<h1>Lead Software Engineer</h1>" + SAVE + APPLY_NEWTAB, cookies=True, chat=True),
    "/job_accept": page("<h1>Engineer</h1>" + APPLY_NEWTAB, cookies=True),
    "/job_sametab": page("<h1>Engineer</h1><a href='/landing'>Apply Now</a>", cookies=True),
    "/landing": page("<h1>Ready to apply?</h1><button onclick=\"location='/form'\">Start application</button>"),
    "/form": page(FORM, chat=True),
    "/job_odd": page("<h1>Engineer</h1><button onclick=\"location='/form'\">Start your journey</button>"
                     "<button>Delete my job alerts</button><a href='/jobs'>Browse all jobs</a>"),
    "/job_none": page("<h1>Engineer</h1><p>Applications are closed.</p><a href='/jobs'>Browse all jobs</a>"),
    "/job_login": page("<h1>Engineer</h1><a href='/login'>Apply</a>"),
    "/login": "<h1>Sign in</h1><label>Email Address <input type=email></label><label>Password <input type=password></label>"
              "<button>Sign In</button><a href='/signup'>Create Account</a>",
    "/job_dialog": page("<h1>Engineer</h1><p>Great role.</p>", extra="""<div role="dialog" aria-modal="true"
        style="position:fixed;inset:60px;background:#fff;z-index:90;padding:20px"><p>We use cookies and need your consent to process data.</p>
        <label for="a">First Name</label><input id="a"><label for="b">Last Name</label><input id="b">
        <label for="c">Email</label><input id="c" type="email"><label for="d">Resume</label><input id="d" type="file">
        <button>Submit application</button></div>"""),
}


class Site:
    def __init__(self):
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = PAGES.get(urlparse(self.path).path, "<p>404</p>")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body.encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


class FakeLLM:
    """Chooses the clickable whose text matches `want`; records what it was shown."""

    def __init__(self, want=None, bogus_id=None):
        self.want, self.bogus_id, self.prompts = want, bogus_id, []

    async def call_tool(self, system, user, tool_name, tool_description, schema):
        self.prompts.append(user)
        items = json.loads(user.split("<clickable_elements>")[1].split("</clickable_elements>")[0])
        if self.bogus_id:
            return {"id": self.bogus_id, "reason": "x"}
        pick = next((i for i in items if i["text"] == self.want), None)
        return {"id": pick["id"] if pick else "none", "reason": "test"}


def with_browser(fn):
    """Run `async fn(site, context, page_factory)` against a fresh headless browser and site."""
    async def go():
        site = Site()
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch()
                context = await browser.new_context(viewport={"width": 1280, "height": 800})
                p = await context.new_page()
                try:
                    return await fn(site, context, p)
                finally:
                    await browser.close()
        finally:
            site.close()
    return asyncio.run(go())


# --------------------------------------------------------------------------- the failure from the screenshot

def test_search_bar_newsletter_and_save_job_are_not_an_application_form():
    async def body(site, context, p):
        await p.goto(site.url + "/job")
        everything = await extract_fields(p, regions=None)
        assert len(everything) >= 5, [f.label for f in everything]  # the fields exist...
        main = await extract_fields(p)
        assert main == [], [(f.label, f.region) for f in main]  # ...but none belong to an application
        assert await looks_like_application_form(p) is False
        regions = {f.label: f.region for f in everything}
        assert regions["Search job title"] == "search" and regions["Location"] == "search"
        assert regions["Enter Email address (Required)"] in {"marketing", "chrome"}
        assert regions["Save job"] == "noise"
        result = await fill_application(p, PROFILE, None, llm=None)
        assert result["rows"] == [], "nothing may be filled on a page that has no application"
    with_browser(body)


def test_real_form_page_ignores_the_chrome_around_it():
    async def body(site, context, p):
        await p.goto(site.url + "/form")
        labels = [f.label for f in await extract_fields(p)]
        assert labels == ["First Name", "Last Name", "Email", "Phone", "Resume/CV"], labels
        assert await looks_like_application_form(p) is True
    with_browser(body)


def test_navigates_past_cookie_dialog_and_chat_panel_to_the_form_in_a_new_tab():
    async def body(site, context, p):
        await p.goto(site.url + "/job")
        nav = await navigate_to_application(context, p, llm=None)
        assert nav.state == "application_form", (nav.state, nav.steps)
        assert nav.page.url.endswith("/form"), "Apply opens in a new tab: it must be followed"
        assert nav.steps[:3] == ["dismissed cookie: 'Reject all'", "dismissed chat/popup: 'Close chat'",
                                 "clicked (rule) 'Apply'"], nav.steps
        assert nav.steps[3:] == ["dismissed chat/popup: 'Close chat'"], "the form page has its own chat panel"
        assert await p.title() == "rejected", "default policy: reject non-essential cookies"
        result = await fill_application(nav.page, PROFILE, None, llm=None)
        labels = [r["label"] for r in result["rows"]]
        assert labels == ["First Name", "Last Name", "Email", "Phone", "Resume/CV"], labels
        assert await nav.page.input_value("#fn") == "Asha" and await nav.page.input_value("#em") == "asha@example.com"
    with_browser(body)


def test_cookie_policy_accept_is_honoured():
    async def body(site, context, p):
        await p.goto(site.url + "/job_accept")
        actions = await dismiss_overlays(p, cookie_policy="accept")
        assert actions == ["cookie: 'Accept cookies'"] and await p.title() == "accepted"
    with_browser(body)


# --------------------------------------------------------------------------- other navigation patterns

def test_same_tab_apply_then_landing_page_then_form():
    async def body(site, context, p):
        await p.goto(site.url + "/job_sametab")
        nav = await navigate_to_application(context, p, llm=None)
        assert nav.state == "application_form" and nav.page.url.endswith("/form"), (nav.state, nav.steps)
        clicks = [s for s in nav.steps if s.startswith("clicked")]
        assert clicks == ["clicked (rule) 'Apply Now'", "clicked (rule) 'Start application'"], clicks
    with_browser(body)


def test_an_application_inside_a_dialog_is_never_dismissed():
    async def body(site, context, p):
        await p.goto(site.url + "/job_dialog")
        nav = await navigate_to_application(context, p, llm=None)
        assert nav.state == "application_form" and nav.steps == [], nav.steps
        assert await p.locator("#a").is_visible(), "the dialog holding the application must stay open"
    with_browser(body)


def test_llm_is_used_only_when_rules_find_nothing_and_only_for_listed_elements():
    async def body(site, context, p):
        await p.goto(site.url + "/job_odd")
        llm = FakeLLM(want="Start your journey")
        nav = await navigate_to_application(context, p, llm=llm)
        assert nav.state == "application_form", (nav.state, nav.steps)
        assert [s for s in nav.steps if s.startswith("clicked")] == ["clicked (llm) 'Start your journey'"], nav.steps
        offered = llm.prompts[0].split("<clickable_elements>")[1]
        assert "Delete my job alerts" not in offered, "dangerous elements are never offered to the LLM"
        assert "Start your journey" in offered and "Browse all jobs" in offered
    with_browser(body)


def test_llm_cannot_pick_an_element_that_was_not_offered():
    async def body(site, context, p):
        await p.goto(site.url + "/job_odd")
        nav = await navigate_to_application(context, p, llm=FakeLLM(bogus_id="9999"))
        assert nav.state == "no_apply_button" and nav.steps == []
    with_browser(body)


def test_no_apply_button_is_reported_not_guessed():
    async def body(site, context, p):
        await p.goto(site.url + "/job_none")
        nav = await navigate_to_application(context, p, llm=None)
        assert nav.state == "no_apply_button"
        nav2 = await navigate_to_application(context, p, llm=FakeLLM(want="nothing matches"))
        assert nav2.state == "no_apply_button"
    with_browser(body)


def test_login_wall_is_handed_back_to_the_caller():
    async def body(site, context, p):
        await p.goto(site.url + "/job_login")
        nav = await navigate_to_application(context, p, llm=None)
        assert nav.state == "login" and nav.page.url.endswith("/login"), (nav.state, nav.page.url)
    with_browser(body)


# --------------------------------------------------------------------------- pure rules

def test_apply_rules():
    def score(label, **kw):
        return apply_score(Clickable(id="0", label=label, **kw))
    for good in ("Apply", "Apply now", "Apply for this job", "Start application", "Easy Apply", "Apply on company site"):
        assert score(good) >= 10, good
    assert score("Sign in to apply") >= 6
    for bad in ("Apply filters", "Applied", "My applications", "How to apply", "Search", "Save job", "Submit application",
                "Delete application", "View all jobs", "Share this job"):
        assert score(bad) == 0, bad


def test_overlay_decisions():
    def ov(text, buttons, fields=0, file=False, attrs=""):
        return {"text": text, "attrs": attrs, "area": 0.3, "nFields": fields, "hasFile": file,
                "buttons": [{"id": str(i), "label": b, "aria": ""} for i, b in enumerate(buttons)]}
    cookie = ov("We use cookies", ["Manage cookies", "Reject all", "Accept cookies"])
    assert decide_overlay_action(cookie, "reject")["label"] == "Reject all"
    assert decide_overlay_action(cookie, "accept")["label"] == "Accept cookies"
    assert decide_overlay_action(ov("We use cookies", ["Manage cookies", "Got it"]), "reject")["label"] == "Got it"
    assert decide_overlay_action(ov("Cookie settings", ["Manage cookies"]), "reject") is None
    chat = {"text": "Hi There! Are you interested?", "attrs": "chat-widget", "area": 0.1, "nFields": 1, "hasFile": False,
            "buttons": [{"id": "0", "label": "", "aria": "Close chat"}]}
    assert decide_overlay_action(chat)["aria"] == "Close chat"
    assert decide_overlay_action(ov("We use cookies, please consent", ["Close"], fields=3)) is None
    assert decide_overlay_action(ov("Upload your resume", ["Close"], file=True)) is None
    assert decide_overlay_action(ov("Welcome back!", ["Close"])) is None, "unknown dialogs are left alone"


def test_noise_filter_does_not_swallow_real_questions():
    """A real question that merely contains a word like 'favourite' is part of the application."""
    html = """<label for="a">Describe your favourite project</label><textarea id="a"></textarea>
    <label for="b">What is your favourite language?</label><input id="b">
    <label><input type="checkbox"> Add to favourites</label><label><input type="checkbox"> Save job</label>
    <label><input type="checkbox"> Remember me</label>"""

    async def body(site, context, p):
        await p.set_content(html)
        main = [f.label for f in await extract_fields(p)]
        assert main == ["Describe your favourite project", "What is your favourite language?"], main
    with_browser(body)


def test_chat_panel_with_unlabelled_icon_buttons_is_closed_via_its_rightmost_icon():
    """Real chat panels often have a header of icon-only buttons: refresh, sound, close."""
    html = """<h1>Job</h1><a href="/form">Apply</a>
    <div class="chat-panel" style="position:fixed;right:10px;top:10px;width:400px;height:300px;background:#ccc;z-index:50">
      <div style="display:flex;justify-content:flex-end;gap:6px;padding:8px">
        <button style="width:28px;height:28px"><svg width="14" height="14"><circle cx="7" cy="7" r="6"/></svg></button>
        <button style="width:28px;height:28px"><svg width="14" height="14"><rect width="12" height="12"/></svg></button>
        <button id="x" style="width:28px;height:28px" onclick="this.closest('.chat-panel').remove()"><svg width="14" height="14"><path d="M0 0L14 14"/></svg></button>
      </div><p>Hi There! Are you interested in this job? Our chat bot can help.</p></div>"""

    async def body(site, context, p):
        await p.set_content(html)
        actions = await dismiss_overlays(p)
        assert actions == ["chat/popup: 'icon button'"], actions
        assert await p.locator(".chat-panel").count() == 0
    with_browser(body)


def test_icon_fallback_only_applies_to_chat_or_popup_panels():
    ov = {"text": "Welcome back, here is your dashboard", "attrs": "", "area": 0.2, "nFields": 0, "hasFile": False,
          "buttons": [{"id": "0", "label": "", "aria": "", "icon": True, "x": 300, "dy": 10}]}
    assert decide_overlay_action(ov) is None, "never click unknown icons on unknown dialogs"
    ov["text"] = "Hi there! Chat with our assistant"
    assert decide_overlay_action(ov)["id"] == "0"