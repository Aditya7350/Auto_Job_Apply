"""Create-or-login handling for job platforms, using one shared credential.

Policy (as configured by the user):
  * Same email + password on every platform, read from the environment.
  * Platform already known in the DB   -> log in.
  * Platform new / login rejected      -> create the account, then log in.
  * Signup says "already exists"       -> account exists with a different password: ask the user.
  * CAPTCHA / OTP / email verification -> NEVER bypassed. Pause and wait for the user.

The password is never logged, never sent to the LLM and never stored in the DB.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable, Optional
from urllib.parse import urlparse

from app.extractor import Field, extract_fields

log = logging.getLogger("auto_job_apply")


class AuthState(str, Enum):
    LOGGED_IN = "logged_in"  # no auth gate visible
    LOGIN = "login"
    SIGNUP = "signup"
    MANUAL = "manual"  # captcha / otp / verify-email: user must act
    UNKNOWN = "unknown"


@dataclass
class Credentials:
    email: str
    password: str = field(repr=False)  # keep out of logs/tracebacks

    @classmethod
    def from_env(cls) -> "Credentials":
        email = os.getenv("JOB_APPLY_EMAIL") or os.getenv("WORKDAY_EMAIL", "")
        password = os.getenv("JOB_APPLY_PASSWORD") or os.getenv("WORKDAY_PASSWORD", "")
        return cls(email.strip(), password)

    @property
    def usable(self) -> bool:
        return bool(self.email and self.password)


@dataclass
class AuthOutcome:
    ok: bool
    status: str  # logged_in | needs_manual | failed | no_credentials
    detail: str = ""


class AccountStore:
    """Remembers which hosts already have an account (per-tenant hosts for Workday)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        with self._con() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS platform_accounts(
                host TEXT PRIMARY KEY, email TEXT, status TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP)""")

    def _con(self):
        return sqlite3.connect(self.db_path)

    def get(self, host: str) -> Optional[str]:
        with self._con() as con:
            row = con.execute("SELECT status FROM platform_accounts WHERE host=?", (host,)).fetchone()
        return row[0] if row else None

    def mark(self, host: str, email: str, status: str) -> None:
        with self._con() as con:
            con.execute(
                "INSERT INTO platform_accounts(host,email,status) VALUES(?,?,?) "
                "ON CONFLICT(host) DO UPDATE SET status=excluded.status, email=excluded.email, "
                "updated_at=CURRENT_TIMESTAMP", (host, email, status))


# --------------------------------------------------------------------------- detection

MANUAL_MARKERS = (
    "i'm not a robot", "i am not a robot", "verify you are human", "verify you're human",
    "solve the captcha", "complete the captcha", "verification code", "one-time", "one time passcode",
    "security code", "verify your email", "check your email", "we sent you", "we've sent", "confirm your email",
    "enter the code", "authenticator app",
)
CAPTCHA_CHALLENGE = 'iframe[title*="recaptcha challenge" i]:visible, iframe[src*="hcaptcha.com/captcha"]:visible'
LOGIN_RX = re.compile(r"^\s*(sign\s*in|log\s*in|login)\s*$", re.I)
SIGNUP_RX = re.compile(r"^\s*(create\s+(an\s+)?account|sign\s*up|register|join\s+now)\s*$", re.I)
LOGIN_SUBMIT_RX = re.compile(r"^\s*(sign\s*in|log\s*in|login|continue|next)\s*$", re.I)
SIGNUP_SUBMIT_RX = re.compile(r"^\s*(create\s+(an\s+)?account|sign\s*up|register|continue|next)\s*$", re.I)
EXISTS_RX = re.compile(r"already (exists?|registered|in use|taken|have an account|been used)|account.*exists", re.I)
REJECT_RX = re.compile(r"invalid|incorrect|not found|no account|doesn.t exist|does not exist|wrong|unrecognized|"
                       r"couldn.t find|try again", re.I)
TERMS_RX = re.compile(r"agree|terms|privacy|consent|acknowledge|accept", re.I)
TERMS_NEEDED = "the signup form needs you to accept terms/privacy yourself"
AGREE_RX = re.compile(r"^\s*(i|yes,? i) (agree|accept|acknowledge|consent)", re.I)


async def _body_text(page) -> str:
    try:
        return (await page.locator("body").inner_text(timeout=3000)).lower()[:15000]
    except Exception:
        return ""


async def _clickables(page):
    """Yield (element, label) for visible, enabled buttons/links in every frame."""
    for frame in list(page.frames):
        try:
            els = frame.locator('button, input[type="submit"], input[type="button"], [role="button"], a')
            for i in range(min(await els.count(), 80)):
                el = els.nth(i)
                try:
                    if not await el.is_visible() or not await el.is_enabled():
                        continue
                    label = (await el.get_attribute("value")) or (await el.inner_text()) or ""
                    label = label.strip() or (await el.get_attribute("aria-label") or "").strip()
                    if label and len(label) < 60:
                        yield el, label
                except Exception:
                    continue
        except Exception:
            continue


async def click_by_label(page, pattern: re.Pattern) -> bool:
    async for el, label in _clickables(page):
        if pattern.match(label):
            try:
                await el.click(timeout=5000)
                return True
            except Exception:
                pass
            try:  # a transparent layer on top (Workday's "click_filter") is the real click target: click through
                await el.click(timeout=3000, force=True)
                return True
            except Exception:
                continue
    return False


async def detect_auth_state(page) -> AuthState:
    body = await _body_text(page)
    if any(m in body for m in MANUAL_MARKERS):
        return AuthState.MANUAL
    try:
        if await page.locator(CAPTCHA_CHALLENGE).count():
            return AuthState.MANUAL
    except Exception:
        pass
    fields = await extract_fields(page, include_passwords=True, regions=None)
    n_pw = sum(1 for f in fields if f.type == "password")
    if n_pw >= 2:
        return AuthState.SIGNUP
    if n_pw == 1:
        labels = [label async for _, label in _clickables(page)]
        has_login = any(LOGIN_RX.match(l) or l.lower() in {"continue", "next"} for l in labels)
        has_signup = any(SIGNUP_RX.match(l) for l in labels)
        return AuthState.SIGNUP if has_signup and not has_login else AuthState.LOGIN
    return AuthState.LOGGED_IN


# --------------------------------------------------------------------------- actions

def _is_email_field(f: Field) -> bool:
    if f.type == "email":
        return True
    text = f"{f.label} {f.name} {f.hint} {f.placeholder}".lower()
    return f.type in {"text", "tel"} and bool(re.search(r"e-?mail|user ?name|login id|username", text))


async def _fill_credentials(page, creds: Credentials, signup: bool) -> Optional[str]:
    """Fill email + password fields. Returns an error string if the form needs the user."""
    fields = await extract_fields(page, include_passwords=True, regions=None)
    email_f = next((f for f in fields if _is_email_field(f)), None)
    pw_fields = [f for f in fields if f.type == "password"]
    if email_f is None or not pw_fields:
        return "could not find email/password fields"
    await email_f.locator().fill(creds.email, timeout=4000)
    for f in pw_fields:  # signup: password + verify password get the same value
        await f.locator().fill(creds.password, timeout=4000)

    if signup:
        auto_accept = os.getenv("AUTO_ACCEPT_TERMS", "false").lower() == "true"
        for f in fields:
            # "I agree to creating this account..." blocks signup even when the site doesn't mark it required
            if f.type == "checkbox" and TERMS_RX.search(f.label) and (f.required or AGREE_RX.search(f.label)):
                if not auto_accept:
                    return TERMS_NEEDED
                await f.locator().set_checked(True, timeout=3000, force=True)
    return None


async def _submit(page, signup: bool) -> bool:
    if await click_by_label(page, SIGNUP_SUBMIT_RX if signup else LOGIN_SUBMIT_RX):
        return True
    try:  # fallback: Enter in a password field
        pw = next((f for f in await extract_fields(page, include_passwords=True, regions=None) if f.type == "password"), None)
        if pw:
            await pw.locator().press("Enter")
            return True
    except Exception:
        pass
    return False


async def _settle(page, was: AuthState, seconds: int = 12) -> AuthState:
    """Wait for the page to react to a submit; return early on any change or visible error."""
    await page.wait_for_timeout(1500)
    waited = 1.5
    while True:
        state = await detect_auth_state(page)
        if state != was:
            return state
        body = await _body_text(page)
        if REJECT_RX.search(body) or EXISTS_RX.search(body) or waited >= seconds:
            return state
        await page.wait_for_timeout(1000)
        waited += 1


async def _wait_for_manual(page, on_manual, timeout_s: int = 300, state: AuthState = AuthState.MANUAL,
                           message: str = "Action needed in the browser: CAPTCHA / verification code / email "
                                          "verification. Complete it there; automation resumes automatically.") -> bool:
    """Tell the user what to do, then wait until the page leaves `state`."""
    if on_manual:
        await on_manual(message)
    waited = 0
    while waited < timeout_s:
        if await detect_auth_state(page) != state:
            return True
        await page.wait_for_timeout(2000)
        waited += 2
    return False


# --------------------------------------------------------------------------- main flow

async def ensure_authenticated(
    page,
    creds: Credentials,
    store: AccountStore,
    on_manual: Optional[Callable[[str], Awaitable[None]]] = None,
) -> AuthOutcome:
    host = (urlparse(page.url).hostname or "").lower()
    state = await detect_auth_state(page)
    if state == AuthState.LOGGED_IN:
        return AuthOutcome(True, "logged_in", "no login gate on this page")
    if not creds.usable:
        return AuthOutcome(False, "no_credentials", "set JOB_APPLY_EMAIL and JOB_APPLY_PASSWORD in .env")

    known = store.get(host) == "active"
    login_attempts = signup_attempts = 0
    signup_submitted = False

    for _ in range(8):  # hard cap: never loop forever
        state = await detect_auth_state(page)
        log.info("AUTH | host=%s state=%s known_account=%s", host, state.value, known)

        if state == AuthState.LOGGED_IN:
            store.mark(host, creds.email, "active")
            return AuthOutcome(True, "logged_in")

        if state == AuthState.MANUAL:
            if not await _wait_for_manual(page, on_manual):
                return AuthOutcome(False, "needs_manual", "timed out waiting for verification")
            continue

        if state == AuthState.LOGIN:
            allowed = 2 if signup_submitted else 1
            if login_attempts >= allowed:
                # Login was rejected. New platform -> the account probably doesn't exist yet.
                if not known and signup_attempts == 0 and await click_by_label(page, SIGNUP_RX):
                    await page.wait_for_timeout(1500)
                    continue
                return AuthOutcome(False, "failed", "login rejected; the account may use a different password")
            login_attempts += 1
            err = await _fill_credentials(page, creds, signup=False)
            if err:
                return AuthOutcome(False, "needs_manual", err)
            if not await _submit(page, signup=False):
                return AuthOutcome(False, "needs_manual", "could not find the Sign In button")
            await _settle(page, AuthState.LOGIN)
            continue

        if state == AuthState.SIGNUP:
            if signup_attempts >= 1:
                body = await _body_text(page)
                if EXISTS_RX.search(body):
                    store.mark(host, creds.email, "active")
                    if login_attempts == 0 and await click_by_label(page, LOGIN_RX):
                        await page.wait_for_timeout(1500)  # never tried logging in yet: do it once
                        continue
                    # Login was already rejected and the account exists: the password differs.
                    # Retrying could lock the account, so stop and let the user fix it.
                    return AuthOutcome(False, "failed",
                                       "an account already exists here with a different password; "
                                       "log in or reset it once manually")
                return AuthOutcome(False, "needs_manual", "account creation did not complete; check the form")
            if known and login_attempts == 0 and await click_by_label(page, LOGIN_RX):
                await page.wait_for_timeout(1500)  # we already have an account here: log in instead
                continue
            signup_attempts += 1
            err = await _fill_credentials(page, creds, signup=True)
            if err == TERMS_NEEDED:  # email + passwords are in; the agreement is the user's own click
                store.mark(host, creds.email, "created")
                if not await _wait_for_manual(page, on_manual, state=AuthState.SIGNUP, message=(
                        "Email and password are filled in. Tick the terms/agreement box and press Create Account "
                        "in the browser; the automation continues by itself.")):
                    return AuthOutcome(False, "needs_manual", "timed out waiting for you to create the account")
                signup_submitted = True
                continue
            if err:
                return AuthOutcome(False, "needs_manual", err)
            if not await _submit(page, signup=True):
                return AuthOutcome(False, "needs_manual", "could not find the Create Account button")
            signup_submitted = True
            store.mark(host, creds.email, "created")
            await _settle(page, AuthState.SIGNUP)
            continue

        return AuthOutcome(False, "needs_manual", "unrecognised login page")

    return AuthOutcome(False, "failed", "too many auth steps")