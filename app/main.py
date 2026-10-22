from pathlib import Path
import json
import asyncio
import re
import sqlite3
import logging
from datetime import datetime
from urllib.parse import urlparse, urljoin
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from playwright.async_api import async_playwright

from app.field_mapper import build_definitions, choose_field, norm
from app.accounts import AccountStore, Credentials, ensure_authenticated
from app.extractor import looks_like_application_form
from app.pipeline import fill_application
from app.review import make_router
from app.llm import get_llm
from app.navigator import navigate_to_application
from app.sessions import SessionManager

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")
PROFILE = BASE / "config" / "profile.json"
DB = BASE / "applications.db"

app = FastAPI(title="Free Auto Job Apply MVP")

LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "automation.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger("auto_job_apply")

def log(message, level="info"):
    getattr(logger, level, logger.info)(message)


LOGIN_URL_MARKERS = (
    "/login", "/signin", "/sign-in", "/authenticate", "/auth/", "login.jsp",
    "signin.jsp", "account/login", "accessmanagement/login"
)
LOGIN_TEXT_MARKERS = (
    "sign in", "sign-in", "log in", "login", "returning user",
    "welcome. you are not signed in", "username", "user name"
)


def load_profile():
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def resolve_resume_path(profile):
    """Resolve the configured resume, with a safe PDF fallback in uploads/."""
    configured = str(profile.get("resume_path", "")).strip()
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = BASE / candidate
        if candidate.exists() and candidate.is_file():
            return candidate

    uploads = BASE / "uploads"
    pdfs = sorted(uploads.glob("*.pdf"), key=lambda x: x.stat().st_mtime, reverse=True)
    if pdfs:
        log(f"Resume fallback selected: {pdfs[0]}")
        return pdfs[0]
    return None


def db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS applications(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT, title TEXT, company TEXT, status TEXT,
        fields_filled INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        notes TEXT
    )""")
    con.commit()
    return con


async def collect_metadata(page, locator):
    data = {}
    for attr in ("name", "id", "placeholder", "aria-label", "autocomplete"):
        try:
            data[attr] = await locator.get_attribute(attr) or ""
        except Exception:
            data[attr] = ""

    try:
        data["tag"] = await locator.evaluate("e => e.tagName.toLowerCase()")
    except Exception:
        data["tag"] = ""

    if data.get("id"):
        try:
            label = page.locator(f'label[for="{data["id"]}"]').first
            if await label.count():
                data["label"] = (await label.inner_text()).strip()
        except Exception:
            pass

    if not data.get("label"):
        try:
            wrapped = locator.locator("xpath=ancestor::label[1]").first
            if await wrapped.count():
                data["label"] = (await wrapped.inner_text()).strip()
        except Exception:
            pass

    if not data.get("label"):
        try:
            data["nearby_text"] = await locator.evaluate("""e => {
                const p = e.parentElement;
                return p ? (p.innerText || '').slice(0, 180) : '';
            }""")
        except Exception:
            data["nearby_text"] = ""

    parts = [data.get(k, "") for k in (
        "label", "aria-label", "placeholder", "name", "id", "autocomplete", "nearby_text"
    )]
    data["metadata"] = " | ".join(x for x in parts if x)
    return data


async def page_text(page):
    try:
        return (await page.locator("body").inner_text(timeout=3000)).lower()[:12000]
    except Exception:
        return ""


async def is_login_gate(page):
    """Detect a login/authentication gate without attempting to bypass it."""
    try:
        current_url = (page.url or "").lower()
        if any(marker in current_url for marker in LOGIN_URL_MARKERS):
            return True

        password = page.locator('input[type="password"]:visible').first
        has_password = await password.count() > 0
        if not has_password:
            return False

        text = await page_text(page)
        marker_hits = sum(1 for marker in LOGIN_TEXT_MARKERS if marker in text)
        return marker_hits >= 1
    except Exception:
        return False


async def wait_for_manual_login(page, timeout_ms=300000):
    """Wait for the user to log in manually; never handles credentials itself."""
    deadline = 0
    interval = 2000
    while deadline < timeout_ms:
        if not await is_login_gate(page):
            return True
        await page.wait_for_timeout(interval)
        deadline += interval
    return False


APPLY_TEXT_MARKERS = (
    "apply now", "apply for this job", "apply for job", "start application",
    "start your application", "begin application", "submit application",
    "apply", "application"
)


def score_apply_candidate(text, href=""):
    """Score visible links/buttons likely to start a job application."""
    t = norm(text or "")
    h = (href or "").lower()
    if not t and not h:
        return -1
    score = 0
    if t in {"apply", "apply now"}:
        score += 100
    elif "apply for this job" in t or "apply for job" in t:
        score += 95
    elif "start application" in t or "begin application" in t:
        score += 90
    elif "submit application" in t:
        score += 85
    elif "application" in t:
        score += 50
    elif "apply" in t:
        score += 60
    if "apply" in h or "application" in h:
        score += 25
    return score


async def find_apply_button(page):
    """Find the most likely application-start control on a job detail page."""
    candidates = page.locator('a, button, input[type="button"], input[type="submit"]')
    best = None
    best_score = 0
    n = await candidates.count()
    for i in range(n):
        el = candidates.nth(i)
        try:
            if not await el.is_visible() or not await el.is_enabled():
                continue
            tag = (await el.evaluate("e => e.tagName.toLowerCase()"))
            if tag == "input":
                text = await el.get_attribute("value") or ""
            else:
                text = (await el.inner_text()).strip()
            aria = await el.get_attribute("aria-label") or ""
            title = await el.get_attribute("title") or ""
            href = await el.get_attribute("href") or ""
            combined = " | ".join(x for x in (text, aria, title) if x)
            score = score_apply_candidate(combined, href)
            if score > best_score:
                best = el
                best_score = score
        except Exception:
            continue
    return best, best_score


async def click_apply_and_get_page(context, page):
    """Open the application in the original tab whenever possible.

    Strategy:
    1. If Apply is a normal link, navigate the current page directly to its URL.
    2. Otherwise click exactly once and observe whether it navigates this tab or
       opens a popup. If a popup is created, reuse its final URL in the original
       tab and close the popup. Fall back to the popup only if same-tab reuse fails.
    """
    button, score = await find_apply_button(page)
    if button is None:
        log("Apply button not found", "warning")
        return page, False, "No likely Apply button found on the current page."

    old_url = page.url

    # A normal href is the cleanest way to avoid opening a second tab.
    try:
        href = (await button.get_attribute("href") or "").strip()
    except Exception:
        href = ""

    if href and not href.lower().startswith(("javascript:", "mailto:", "tel:")) and href != "#":
        destination = urljoin(page.url, href)
        try:
            log(f"Apply link found; navigating in the same tab: {destination}")
            await page.goto(destination, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(1200)
            return page, True, f"Clicked Apply link (score={score}) using the same tab."
        except Exception as exc:
            log(f"Direct same-tab Apply navigation failed; trying click fallback: {exc}", "warning")

    # For JS-driven controls, click once and detect popup vs same-tab navigation.
    popup = None
    try:
        async with context.expect_page(timeout=5000) as page_info:
            await button.click(timeout=10000)
        popup = await page_info.value
    except Exception:
        # Timeout is expected for ordinary same-tab navigation. Do not click again:
        # a second click could submit or launch the application twice.
        popup = None

    if popup is not None:
        try:
            await popup.wait_for_load_state("domcontentloaded", timeout=20000)
        except Exception:
            pass
        await popup.wait_for_timeout(1000)
        popup_url = popup.url
        if popup_url and popup_url != "about:blank":
            try:
                log(f"Apply opened a popup; reusing URL in original tab: {popup_url}")
                await page.goto(popup_url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(1200)
                try:
                    await popup.close()
                except Exception:
                    pass
                return page, True, f"Apply popup redirected into the original tab (score={score})."
            except Exception as exc:
                log(f"Could not reuse popup URL in original tab: {exc}", "warning")
        # Some sites depend on popup-specific state; retain it as a safe fallback.
        log("Keeping popup because same-tab reuse was not possible", "warning")
        return popup, True, f"Apply opened a new tab; same-tab reuse was not possible (score={score})."

    try:
        await page.wait_for_load_state("domcontentloaded", timeout=15000)
    except Exception:
        pass
    await page.wait_for_timeout(1200)
    if page.url != old_url:
        log(f"Apply navigated in the original tab: {page.url}")
        return page, True, f"Clicked Apply (score={score}) and navigated in the same tab."

    # JS may update the page without changing the URL; inspect the current page.
    log(f"Apply control clicked; staying on current page: {page.url}")
    return page, True, f"Clicked Apply (score={score}); checking the current page for the application."

ATS_PAGE_MARKERS = (
    ("Workday", ("myworkdaycdn", "myworkdayjobs", "workday, inc", "data-automation-id")),
    ("Greenhouse", ("boards.greenhouse.io", "grnhse", "greenhouse.io/embed")),
    ("Lever", ("jobs.lever.co", "lever.co/")),
    ("iCIMS", ("icims.com",)),
    ("SmartRecruiters", ("smartrecruiters",)),
    ("Taleo", ("taleo.net",)),
    ("SuccessFactors", ("successfactors", "sap.com/career")),
    ("Oracle Recruiting", ("oraclecloud.com", "candidateexperience")),
    ("Ashby", ("ashbyhq",)),
    ("Phenom", ("phenompeople", "phenom.com")),
)

async def detect_ats_from_page(page) -> str:
    blobs = [page.url.lower()] + [f.url.lower() for f in page.frames]
    try:
        blobs.append((await page.content()).lower()[:600000])
    except Exception:
        pass
    text = " ".join(blobs)
    for name, markers in ATS_PAGE_MARKERS:
        if any(m in text for m in markers):
            return name
    return "Unknown"

async def detect_ats(page):
    host = (urlparse(page.url).hostname or "").lower()
    if "myworkdayjobs.com" in host or "workday" in host:
        return "Workday"
    if "greenhouse" in host:
        return "Greenhouse"
    if "lever.co" in host or "jobs.lever" in host:
        return "Lever"
    if "taleo" in host:
        return "Taleo"
    for marker, name in (("icims", "iCIMS"), ("smartrecruiters", "SmartRecruiters"), ("ashbyhq", "Ashby"),
                         ("bamboohr", "BambooHR"), ("successfactors", "SuccessFactors"), ("jobvite", "Jobvite"),
                         ("oraclecloud", "Oracle Recruiting"), ("phenompeople", "Phenom")):
        if marker in host:
            return name
    if "jobseqno=" in page.url.lower():  # Phenom-hosted company career sites (careers.mastercard.com...)
        return "Phenom"
    return await detect_ats_from_page(page)

async def save_debug(page, label):
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", label)
        path = LOG_DIR / f"{ts}_{safe}.png"
        await page.screenshot(path=str(path), full_page=True)
        log(f"Debug screenshot saved: {path}")
        return str(path)
    except Exception as exc:
        log(f"Could not save screenshot: {exc}", "warning")
        return ""


async def inspect_application_fields(page):
    """Log visible application controls in the main document and child frames.

    This diagnostic deliberately logs field metadata only, never entered values.
    """
    frames = list(page.frames)
    log(f"FORM INSPECT START | frames={len(frames)}")
    for frame_index, frame in enumerate(frames):
        try:
            controls = frame.locator(
                'input:not([type="hidden"]), textarea, select, '
                '[role="textbox"], [role="combobox"]'
            )
            count = await controls.count()
            log(
                f"FORM INSPECT | frame={frame_index} "
                f"| url={frame.url} | controls={count}"
            )
            for i in range(min(count, 100)):
                el = controls.nth(i)
                try:
                    if not await el.is_visible():
                        continue
                    metadata = {}
                    for attr in (
                        "type", "name", "id", "placeholder", "aria-label",
                        "autocomplete", "data-automation-id", "role"
                    ):
                        metadata[attr] = await el.get_attribute(attr) or ""
                    tag = await el.evaluate("e => e.tagName.toLowerCase()")
                    try:
                        label = await el.evaluate("""e => {
                            const id = e.id;
                            const linked = id ? document.querySelector(
                                'label[for="' + CSS.escape(id) + '"]'
                            ) : null;
                            const wrapped = e.closest('label');
                            const labelledBy = e.getAttribute('aria-labelledby');
                            const ariaLabelled = labelledBy
                                ? document.getElementById(labelledBy)
                                : null;
                            return ((linked && linked.innerText) ||
                                    (wrapped && wrapped.innerText) ||
                                    (ariaLabelled && ariaLabelled.innerText) ||
                                    '').trim();
                        }""")
                    except Exception:
                        label = ""
                    log(
                        f"FORM CONTROL | frame={frame_index} | index={i} "
                        f"| tag={tag} | label={label!r} | metadata={metadata}"
                    )
                except Exception as exc:
                    log(f"FORM CONTROL inspection error | frame={frame_index} index={i} | {exc}", "warning")
        except Exception as exc:
            log(f"FORM FRAME inspection error | frame={frame_index} | {exc}", "warning")
    log("FORM INSPECT FINISHED")


async def fill_page(page):
    """Fill profile-matched controls across the main document and child frames."""
    profile = load_profile()
    definitions = build_definitions(profile)
    filled = 0
    matches = []
    seen = set()
    frames = list(page.frames)
    log(f"FIELD MAPPING START | frames={len(frames)}")

    for frame_index, frame in enumerate(frames):
        controls = frame.locator(
            'input:not([type="hidden"]), textarea, select, '
            '[role="textbox"], [role="combobox"]'
        )
        try:
            count = await controls.count()
        except Exception as exc:
            log(f"FIELD FRAME unavailable | frame={frame_index} | {exc}", "warning")
            continue
        log(f"FIELD FRAME | frame={frame_index} | controls={count} | url={frame.url}")

        for i in range(count):
            el = controls.nth(i)
            try:
                if not await el.is_visible() or not await el.is_enabled():
                    continue
                tag = (await el.evaluate("e => e.tagName.toLowerCase()")).lower()
                typ = (await el.get_attribute("type") or "text").lower()
                # Avoid filling the same DOM element twice if a locator strategy overlaps.
                key = (frame_index, i, tag, typ)
                if key in seen:
                    continue
                seen.add(key)
                meta = await collect_metadata(frame, el)

                if typ == "file":
                    resume = resolve_resume_path(profile)
                    if resume is not None:
                        await el.set_input_files(str(resume))
                        filled += 1
                        matches.append({"field": "resume", "metadata": meta["metadata"], "score": 100})
                        log(f"FIELD | resume | frame={frame_index} | uploaded={resume}")
                    else:
                        log(f"FIELD | resume | no PDF found in {BASE / 'uploads'}", "warning")
                    continue

                if typ in {"checkbox", "radio", "submit", "button", "reset", "image", "password", "hidden"}:
                    continue

                definition = choose_field(meta.get("metadata", ""), definitions)
                if not definition:
                    continue
                value = str(definition.value)
                if not value.strip():
                    continue

                if tag == "select":
                    selected = False
                    try:
                        await el.select_option(label=value)
                        selected = True
                    except Exception:
                        try:
                            await el.select_option(value=value)
                            selected = True
                        except Exception:
                            options = el.locator("option")
                            for j in range(await options.count()):
                                option = options.nth(j)
                                option_text = await option.inner_text()
                                option_value = await option.get_attribute("value")
                                if option_value is not None and (norm(option_text) == norm(value) or norm(value) in norm(option_text)):
                                    await el.select_option(value=option_value)
                                    selected = True
                                    break
                    if not selected:
                        log(f"FIELD SKIP | select option not matched | {meta.get('metadata', '')}")
                        continue
                else:
                    await el.fill(value)

                filled += 1
                matches.append({"field": definition.key, "metadata": meta.get("metadata", ""), "score": "matched"})
                log(f"FIELD FILLED | {definition.key} | frame={frame_index} | metadata={meta.get('metadata', '')}")
            except Exception as exc:
                log(f"FIELD CONTROL error | frame={frame_index} index={i} | {type(exc).__name__}: {exc}", "warning")

    log(f"FIELD MAPPING END | filled={filled} | matched_fields={len(matches)}")
    return filled, matches


@app.get("/", response_class=HTMLResponse)
async def home():
    return '''<!doctype html>
<html><head><meta charset="utf-8"><title>Auto Job Apply</title>
<style>body{font-family:Arial;max-width:900px;margin:50px auto;padding:0 20px}input{width:100%;padding:12px;box-sizing:border-box}button{padding:12px 20px;cursor:pointer}.card{border:1px solid #ddd;padding:20px;border-radius:10px}.note{background:#f5f7fa;padding:12px;border-radius:8px}</style>
</head><body><div class="card"><h1>Free Auto Job Apply MVP</h1>
<p>Paste a job application URL. Playwright opens it, detects login gates, intelligently maps fields to your profile and uploads your resume.</p>
<form method="post" action="/apply"><input name="url" placeholder="https://..." required><br><br><button>Open & Auto-Fill</button></form>
<p class="note"><strong>Login:</strong> uses the email/password from your .env (creates the account on new sites, logs in on known ones). <strong>CAPTCHA, OTP and email verification</strong> are never bypassed: the browser pauses and waits for you. After filling, you review everything on the next page; <strong>nothing is submitted until you press Approve.</strong></p>
</div></body></html>'''


# ======================================================================================
# Sessions: the browser stays open until YOU approve, cancel, or the idle timeout hits.
# ======================================================================================
CREDS = Credentials.from_env()
if CREDS.usable:  # one credential for everything: the Workday code reads these names
    if not os.getenv("WORKDAY_EMAIL"):
        os.environ["WORKDAY_EMAIL"] = CREDS.email
    if not os.getenv("WORKDAY_PASSWORD"):
        os.environ["WORKDAY_PASSWORD"] = CREDS.password
ACCOUNTS = AccountStore(DB)


def persist_session(s):
    con = db()
    con.execute("UPDATE applications SET status=?, title=?, fields_filled=?, notes=? WHERE id=?",
                (s.status, s.title, s.summary.get("filled", 0),
                 (s.message or "")[:900] + (f" Screenshot: {s.screenshot}" if s.screenshot else ""), s.id))
    con.commit()
    con.close()


WORKDAY_START_RX = (re.compile(r"^\s*autofill with resume\s*$", re.I), re.compile(r"^\s*apply manually\s*$", re.I))


async def start_workday_application(page) -> bool:
    """Workday's 'Start Your Application' page: press 'Autofill with Resume' (else 'Apply Manually') and wait
    for the next screen. Account creation/sign-in that follows is the generic flow (app.accounts), like
    every other site; the resume upload page after it is an ordinary form page."""
    for rx in WORKDAY_START_RX:
        button = page.get_by_role("button", name=rx)
        try:
            await button.first.wait_for(state="visible", timeout=8000)
        except Exception:
            continue
        log(f"Workday: pressing {await button.first.inner_text()!r}")
        await button.first.click(timeout=10000)
        try:  # the next screen (account form, upload page, form) renders after a loading page
            await page.locator('input[type="password"]:visible, input[type="file"], '
                               '[data-automation-id="progressBarActiveStep"]').first.wait_for(timeout=20000)
        except Exception:
            pass
        await page.wait_for_timeout(1500)
        return True
    return False

def save_answers(answers: list) -> int:
    p = load_profile()
    custom = p.setdefault("custom_answers", [])
    saved = 0
    for a in answers:
        label, value = str(a.get("label", "")).strip(), str(a.get("value", "")).strip()
        if not label or not value:
            continue
        custom[:] = [c for c in custom if norm(c.get("question_pattern", "")) != norm(label)]
        custom.append({"question_pattern": label, "answer": value})
        saved += 1
    PROFILE.write_text(json.dumps(p, indent=2, ensure_ascii=False), encoding="utf-8")
    return saved

async def _run_job(s):
    profile = load_profile()
    resume_path = resolve_resume_path(profile)
    job = {"url": s.url}

    async def notify_manual(message):
        s.set("login_required", message)

    async def authenticate(page, ats):
        """Generic create-or-login, the same for every site (Workday included)."""
        if not await is_login_gate(page):
            return True
        s.set("login_required", "Login page detected: signing in (or creating the account)...")
        outcome = await ensure_authenticated(page, CREDS, ACCOUNTS, on_manual=notify_manual)
        if not outcome.ok:
            s.set("login_required", outcome.detail or outcome.status)
        return outcome.ok

    async def refill():
        """Re-scan whatever page the user is on now and fill it. Never overwrites existing values."""
        pages = [p for p in s.context.pages if not p.is_closed()]
        if pages:
            s.page = pages[-1]
        page = s.page
        await page.bring_to_front()
        ats = await detect_ats(page)
        if not await authenticate(page, ats):
            return {"rows": s.rows, "filled": 0, "needs_review": 0, "required_unfilled": []}
        job["url"], job["ats"] = page.url, ats
        return await fill_application(page, load_profile(), resume_path, job=job)

    s.refill = refill

    s.set("opening", "Opening the job page...")
    page = s.page
    await page.goto(s.url, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(1500)
    s.title = await page.title()
    job["title"] = s.title
    job["description"] = (await page_text(page))[:3500]  # context for the LLM's free-text answers
    log(f"Job page loaded | title={s.title!r} | url={page.url}")
    body_text = (await page_text(page))[:3000]
    if re.search(r"no longer available|position (has been )?(closed|filled)|job (has expired|not found)|page (you are|you're) looking for", body_text, re.I):
        await save_debug(page, "job_closed")
        s.set("job_closed", "This posting is closed or moved, so I stopped instead of picking a different job. " "Paste the URL of a live posting.")
        return
    asked, landed = urlparse(s.url), urlparse(page.url)
    redirect_note = ""
    if (asked.hostname, asked.path.rstrip("/")) != (landed.hostname, landed.path.rstrip("/")):
        redirect_note = f" The job link redirected to {page.url}: the posting may be closed or moved."
        log(f"Job link redirected | {s.url} -> {page.url}", "warning")

    def live_page(preferred):
        """The tab to work in: a site can close the one we had (a popup flow, a redirect)."""
        if preferred is not None and not preferred.is_closed():
            return preferred
        pages = [p for p in s.context.pages if not p.is_closed()]
        if not pages:
            raise RuntimeError("The browser window was closed.")
        return pages[-1]

    async def workday_ready(p):
        """Hand over to the Workday state machine once we are past the job description."""
        if await detect_ats(p) != "Workday":
            return False
        txt = (await page_text(p)).lower()
        return "/apply" in p.url.lower() or any(k in txt for k in ("autofill with resume", "apply manually",
                                                                    "use my last application"))

    llm = get_llm()  # optional: only used when the rules cannot find the Apply control
    nav = None
    for _ in range(3):  # navigate -> (sign in) -> navigate again
        s.set("opening", "Looking for the application form...")
        nav = await navigate_to_application(s.context, page, llm=llm, stop_when=workday_ready)
        page = s.page = live_page(nav.page)
        for step in nav.steps:
            log(f"NAV | {step}")
        s.title = await page.title()
        if nav.state != "login":
            break
        if not await authenticate(page, await detect_ats(page)):
            await save_debug(page, "login_failed")
            return  # status/message already set; the user can log in by hand and press 'Re-scan & fill'
    if nav.state not in {"application_form", "custom"}:
        await save_debug(page, nav.state)
        s.set("apply_button_not_found",
              "Could not find the application form or an Apply button on this site. Open the application page "
              "in the browser yourself, then press 'Re-scan & fill'." + redirect_note +
              " Steps tried: " + "; ".join(nav.steps)[:300])
        return

    ats = await detect_ats(page)
    job["ats"] = ats
    log(f"ATS detected: {ats}")

    if ats == "Workday" and await start_workday_application(page):
        log(f"Workday: next screen | url={page.url}")
    if not await authenticate(page, ats):
        await save_debug(page, "login_failed")
        return  # status/message already set; the user can log in by hand and press 'Re-scan & fill'

    s.set("filling", "Reading the form and filling it...")
    s.step_name = await manager._step_name(page)
    result = await fill_application(page, load_profile(), resume_path, job=job)
    s.apply_result(result)
    if not result["rows"]:
        s.screenshot = await save_debug(page, "no_application_form")
        s.set("no_application_form", "No application fields were found on this page. Navigate to the form in the "
                                     "browser, then press 'Re-scan & fill'.")
        return
    s.set("filling", "Page 1 filled. Moving through the next pages...")
    outcome = await manager._advance_steps(s)
    if not s.final_reached:
        s.screenshot = await save_debug(s.page, "ready_for_review")
    s.set(outcome["status"], f"ATS={ats}. {s.summary.get('filled', 0)} fields filled on {s.step} page(s), "
                             f"{s.summary.get('needs_review', 0)} need your review. {outcome['message']} "
                             "Nothing has been submitted.")


async def run_job(s):
    try:
        await _run_job(s)
    except asyncio.CancelledError:
        raise
    except Exception:
        if s.page is not None:
            await save_debug(s.page, "error")
        raise


manager = SessionManager(
    run_job, persist=persist_session, screenshot_fn=save_debug,
    headless=os.getenv("HEADLESS", "").lower() in {"1", "true", "yes"},
)
app.include_router(make_router(manager))


@app.on_event("shutdown")
async def _close_sessions():
    await manager.shutdown()


@app.post("/apply")
async def apply(url: str = Form(...)):
    if urlparse(url).scheme not in {"http", "https"}:
        return JSONResponse({"error": "Invalid URL"}, status_code=400)
    log("=" * 70)
    log(f"JOB START | URL: {url}")
    con = db()
    sid = con.execute("INSERT INTO applications(url,status,fields_filled,notes) VALUES(?,?,?,?)",
                      (url, "starting", 0, "")).lastrowid
    con.commit()
    con.close()
    try:
        s = await manager.start(url, sid)
    except RuntimeError as exc:
        con = db()
        con.execute("UPDATE applications SET status='rejected', notes=? WHERE id=?", (str(exc), sid))
        con.commit()
        con.close()
        return JSONResponse({"error": str(exc)}, status_code=429)
    return RedirectResponse(f"/review/{sid}?t={s.token}", status_code=303)


@app.get("/applications")
def applications():
    con = db()
    rows = con.execute("SELECT id,url,title,status,fields_filled,created_at,notes FROM applications ORDER BY id DESC").fetchall()
    con.close()
    keys = ["id", "url", "title", "status", "fields_filled", "created_at", "notes"]
    return [dict(zip(keys, row)) for row in rows]