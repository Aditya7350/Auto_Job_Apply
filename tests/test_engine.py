"""Offline tests. No network, no real LLM: the LLM is replaced by a small fake (plus mocked HTTP for the SDK adapters).

Run:  pytest -q tests/test_engine.py
"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.async_api import async_playwright  # noqa: E402

from app.accounts import AccountStore, Credentials, ensure_authenticated  # noqa: E402
from app.extractor import extract_fields  # noqa: E402
from app.filler import apply_plan  # noqa: E402
from app.planner import plan_fields  # noqa: E402

FORM_HTML = """
<form>
  <label for="fn">First name *</label><input id="fn" name="first_name" required>
  <label for="ln">Last name</label><input id="ln" name="last_name">
  <label>Email address * <input type="email" name="email"></label>
  <label for="ph">Phone</label><input id="ph" type="tel">
  <label for="li">LinkedIn URL</label><input id="li">
  <label for="co">Country</label>
  <select id="co"><option value="">Select...</option><option>United States</option><option>India</option></select>

  <label for="city">Current city</label>
  <input id="city" role="combobox" aria-autocomplete="list" autocomplete="off">
  <ul id="opts" role="listbox" style="display:none"></ul>

  <fieldset><legend>Do you have experience with Python?</legend>
    <label><input type="radio" name="auth" value="y"> Yes</label>
    <label><input type="radio" name="auth" value="n"> No</label></fieldset>

  <fieldset><legend>Will you now or in the future require sponsorship?</legend>
    <label><input type="radio" name="spons" value="y"> Yes</label>
    <label><input type="radio" name="spons" value="n"> No</label></fieldset>

  <fieldset><legend>Gender</legend>
    <label class="fancy"><input type="radio" name="g" value="m" style="opacity:0;position:absolute"> Male</label>
    <label class="fancy"><input type="radio" name="g" value="f" style="opacity:0;position:absolute"> Female</label></fieldset>

  <label for="why">Why do you want to work here?</label><textarea id="why" maxlength="400"></textarea>
  <label for="ctc">Expected CTC</label><input id="ctc">
  <label><input type="checkbox" name="priv" required> I agree to the privacy policy</label>

  <label for="cv">Resume/CV</label>
  <input id="cv" type="file" style="opacity:0;position:absolute;width:1px;height:1px">
  <label for="cl">Cover letter</label><input id="cl" type="file">
  <input type="password" name="pw">
  <button type="submit">Submit application</button>
</form>
<iframe id="fr" srcdoc='<label for="gh">GitHub profile URL</label><input id="gh">'></iframe>
<script>
  const cities = ["Pune", "Mumbai", "Pune Camp", "Delhi"];
  const inp = document.getElementById('city'), ul = document.getElementById('opts');
  inp.addEventListener('input', () => {
    const v = inp.value.toLowerCase();
    ul.innerHTML = '';
    cities.filter(c => c.toLowerCase().startsWith(v)).forEach(c => {
      const li = document.createElement('li'); li.setAttribute('role', 'option'); li.textContent = c;
      li.onclick = () => { inp.value = c; ul.style.display = 'none'; };
      ul.appendChild(li);
    });
    ul.style.display = v ? 'block' : 'none';
  });
</script>
"""

PROFILE = {
    "first_name": "Asha", "last_name": "Rao", "full_name": "Asha Rao", "email": "asha@example.com",
    "phone": "+91 90000 00000", "city": "Pune", "state": "Maharashtra", "country": "India",
    "linkedin": "https://www.linkedin.com/in/asha", "github": "https://github.com/asha",
    "years_experience": "1", "current_title": "AI Engineer", "notice_period": "30 days",
    "custom_answers": [{"question_pattern": "require sponsorship", "answer": "No"}],
}


class FakeLLM:
    """Implements the app.llm adapter interface (call_tool) with canned answers that cite evidence."""

    def __init__(self):
        self.calls = []

    async def call_tool(self, system, user, tool_name, tool_description, schema):
        self.calls.append(user)
        fields = json.loads(user.split("<form_fields>")[1].split("</form_fields>")[0])
        answers = []
        for f in fields:
            lab = f["label"].lower()
            if "python" in lab:
                answers.append({"id": f["id"], "action": "check", "value": "Yes", "confidence": 0.92,
                                "evidence": "years experience: 1"})
            elif "why do you want" in lab:
                answers.append({"id": f["id"], "action": "fill", "value": "I build LLM apps and want to grow here.",
                                "confidence": 0.8, "evidence": "current title: AI Engineer"})
            else:  # anything else is not in the profile
                answers.append({"id": f["id"], "action": "skip", "value": "", "confidence": 0, "evidence": ""})
        return {"answers": answers}


def run(coro):
    return asyncio.run(coro)


def test_extract_plan_fill():
    async def go():
        tmp = Path(tempfile.mkdtemp())
        resume = tmp / "resume.pdf"
        import pymupdf
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), "Asha Rao - AI Engineer. Built LLM apps with Python and FastAPI.")
        doc.save(resume)

        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(FORM_HTML)
            await page.wait_for_timeout(300)

            fields = await extract_fields(page)
            by_label = {f.label: f for f in fields}
            assert by_label["First name"].required, "required star should be detected and stripped"
            assert by_label["Country"].type == "select" and len(by_label["Country"].options) == 3
            radios = [f for f in fields if f.type == "radio"]
            assert len(radios) == 3 and all(len(r.options) == 2 for r in radios), "radios must group by name"
            assert any(f.label == "GitHub profile URL" for f in fields), "iframe field must be found"
            assert not any(f.type == "password" for f in fields), "passwords never go to the planner"

            llm = FakeLLM()
            plan = await plan_fields(fields, PROFILE, resume_path=resume, llm=llm)
            sources = {a.label: a.source for a in plan}
            assert sources["First name"] == "profile" and sources["Country"] == "profile"
            assert sources["Will you now or in the future require sponsorship?"] == "custom"
            assert sources["Do you have experience with Python?"] == "llm"
            assert len(llm.calls) == 1, "one batched LLM call, not one per field"
            sent = llm.calls[0]
            assert "First name" not in sent, "profile-resolved fields must not be sent to the LLM"
            assert "asha@example.com" in sent, "profile facts are sent as grounding"

            actions = {a.label: a for a in plan}
            assert actions["Why do you want to work here?"].needs_review, "AI-written text is always reviewed"
            assert actions["Expected CTC"].action == "skip"
            assert actions["Gender"].action == "skip"
            consent = actions["I agree to the privacy policy"]
            assert consent.action == "skip" and "answer this yourself" in consent.reason and consent.needs_review
            for sensitive in ("privacy policy", "Gender", "Expected CTC"):
                assert sensitive not in sent, f"{sensitive!r} must never be sent to the LLM"
            assert actions["Do you have experience with Python?"].needs_review, "every AI answer is flagged"
            assert actions["Cover letter"].action == "skip", "only the resume field gets the resume"

            results = await apply_plan(page, fields, plan)
            bad = [r for r in results if r.status in {"error", "verified_failed"}]
            assert not bad, bad

            v = lambda sel: page.input_value(sel)  # noqa: E731
            assert await v("#fn") == "Asha" and await v("#ln") == "Rao"
            assert await page.input_value('input[name="email"]') == "asha@example.com"
            assert await v("#co") == "India"
            assert await v("#city") == "Pune", "combobox should pick the exact option, not 'Pune Camp'"
            assert await page.is_checked('input[name="auth"][value="y"]')
            assert await page.is_checked('input[name="spons"][value="n"]')
            assert not await page.is_checked('input[name="priv"]'), "the LLM must never tick consent boxes"
            assert not await page.is_checked('input[name="g"][value="m"]')
            assert await v("#ctc") == ""
            assert "LLM apps" in await v("#why")
            assert await page.evaluate("document.getElementById('cv').files.length") == 1
            assert await page.evaluate("document.getElementById('cl').files.length") == 0
            frame = page.frames[1]
            assert await frame.input_value("#gh") == "https://github.com/asha"
            await browser.close()
    run(go())


# ------------------------------------------------------------------ account flows

LOGIN = """<h1>Sign in</h1><div id=err></div>
<label>Email Address <input type=email id=e></label><label>Password <input type=password id=p></label>
<button onclick="go()">Sign In</button> <a href="/signup">Create Account</a>
<script>function go(){const a=JSON.parse(localStorage.accts||'{}');
 if(a[e.value]===p.value) location='/apply'; else err.textContent='Invalid email or password';}</script>"""

SIGNUP_AUTO = """<h1>Create your account</h1><div id=err></div>
<label>Email Address <input id=e></label><label>Password <input type=password id=p></label>
<label>Verify New Password <input type=password id=v></label>
<button onclick="mk()">Create Account</button> <a href="/login">Sign In</a>
<script>function mk(){const a=JSON.parse(localStorage.accts||'{}');
 if(a[e.value]){err.textContent='An account already exists for this email';return}
 if(p.value!==v.value){err.textContent='Passwords differ';return}
 a[e.value]=p.value;localStorage.accts=JSON.stringify(a);location='%s';}</script>"""

APPLY = "<h1>Application</h1><label>First name <input id=fn></label>"


def _site(signup_next):
    pages = {"/login": LOGIN, "/signup": SIGNUP_AUTO % signup_next, "/apply": APPLY}

    async def handler(route):
        path = "/" + route.request.url.split("/", 3)[3].split("?")[0]
        await route.fulfill(status=200, content_type="text/html", body=pages.get(path, "<p>404</p>"))
    return handler


def _auth(signup_next, preseed=None, creds=None):
    async def go():
        db = Path(tempfile.mkdtemp()) / "t.db"
        store = AccountStore(db)
        creds_ = creds or Credentials("asha@example.com", "S3cret!pass")
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            ctx = await browser.new_context()
            await ctx.route("https://jobs.example.test/**", _site(signup_next))
            page = await ctx.new_page()
            await page.goto("https://jobs.example.test/login")
            if preseed:
                await page.evaluate("v => localStorage.accts = v", json.dumps(preseed))
            out = await ensure_authenticated(page, creds_, store)
            url = page.url
            await browser.close()
        return out, url, store.get("jobs.example.test")
    return run(go())


def test_new_platform_creates_account_then_continues():
    out, url, status = _auth("/apply")
    assert out.ok and url.endswith("/apply") and status == "active"


def test_new_platform_signup_redirects_back_to_login():
    out, url, status = _auth("/login")
    assert out.ok and url.endswith("/apply") and status == "active"


def test_existing_account_just_logs_in():
    out, url, _ = _auth("/apply", preseed={"asha@example.com": "S3cret!pass"})
    assert out.ok and url.endswith("/apply")


def test_existing_account_with_other_password_asks_user():
    out, url, _ = _auth("/apply", preseed={"asha@example.com": "different"})
    assert not out.ok and out.status in {"failed", "needs_manual"}


def test_no_credentials_is_reported_not_guessed():
    out, _, _ = _auth("/apply", creds=Credentials("", ""))
    assert not out.ok and out.status == "no_credentials"


# ------------------------------------------------------------------ provider adapters (real SDKs, mocked HTTP)

import httpx  # noqa: E402

SCHEMA = {"type": "object", "properties": {"answers": {"type": "array", "items": {"type": "object"}}},
          "required": ["answers"]}
ANSWERS = {"answers": [{"id": "0:1", "action": "fill", "value": "Pune", "confidence": 0.9}]}


def test_openai_adapter_sends_forced_function_call_and_parses_reply():
    from openai import AsyncOpenAI
    from app.llm import OpenAILLM
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {
                    "name": "submit_answers", "arguments": json.dumps(ANSWERS)}}]}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    sdk = AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    out = run(OpenAILLM(client=sdk, model="gpt-test").call_tool("sys", "usr", "submit_answers", "d", SCHEMA))
    assert out == ANSWERS
    b = seen["body"]
    assert b["model"] == "gpt-test"
    assert b["tool_choice"] == {"type": "function", "function": {"name": "submit_answers"}}
    assert b["tools"][0]["function"]["parameters"] == SCHEMA
    assert [m["role"] for m in b["messages"]] == ["system", "user"]


def test_openai_adapter_survives_malformed_arguments():
    from openai import AsyncOpenAI
    from app.llm import OpenAILLM

    def handler(request):
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "x", "type": "function", "function": {
                    "name": "submit_answers", "arguments": "{not json"}}]}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    sdk = AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert run(OpenAILLM(client=sdk).call_tool("s", "u", "submit_answers", "d", SCHEMA)) == {}


def test_anthropic_adapter_sends_forced_tool_call_and_parses_reply():
    """Duck-typed client: independent of the installed anthropic SDK's HTTP internals."""
    from app.llm import AnthropicLLM
    seen = {}

    class FakeMessages:
        async def create(self, **kw):
            seen.update(kw)
            block = SimpleNamespace(type="tool_use", name="submit_answers", input=ANSWERS)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok"), block])

    out = run(AnthropicLLM(client=SimpleNamespace(messages=FakeMessages()), model="claude-test")
              .call_tool("sys", "usr", "submit_answers", "d", SCHEMA))
    assert out == ANSWERS
    assert seen["model"] == "claude-test" and seen["system"] == "sys"
    assert seen["tool_choice"] == {"type": "tool", "name": "submit_answers"}
    assert seen["tools"][0]["input_schema"] == SCHEMA
    assert seen["messages"] == [{"role": "user", "content": "usr"}]


def test_provider_selected_by_env(monkeypatch):
    from app.llm import AnthropicLLM, OpenAILLM, get_llm
    for k in ("LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert get_llm() is None
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert isinstance(get_llm(), OpenAILLM)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert isinstance(get_llm(), AnthropicLLM)


# ------------------------------------------------------------------ regressions from real runs

def test_option_matching_uses_whole_words():
    """Real bug: 'India' was matched to 'British Indian Ocean Territory +246' (phone country codes)."""
    from app.extractor import Option
    from app.planner import match_option
    phone = [Option(label=x) for x in ("British Indian Ocean Territory +246", "Indonesia +62", "India +91",
                                       "Indiana (US) +1")]
    assert match_option("India", phone).label == "India +91"
    assert match_option("Indonesia", phone).label == "Indonesia +62"
    assert match_option("Ocean", [Option(label="Pacific Ocean Territory")]).label == "Pacific Ocean Territory"
    assert match_option("land", [Option(label="Iceland"), Option(label="Ireland")]) is None, "no raw substrings"
    yn = [Option(label="Yes"), Option(label="No")]
    assert match_option("true", yn).label == "Yes" and match_option("No", yn).label == "No"


def test_evidence_must_exist_in_profile_or_resume():
    from app.planner import build_corpus, evidence_supported
    corpus = build_corpus({**PROFILE, "facts": {"willing_to_relocate": "Yes"}}, "Built RAG apps with FastAPI.")
    assert evidence_supported("willing to relocate: Yes", corpus)
    assert evidence_supported("current title: AI Engineer", corpus)
    assert evidence_supported("Built RAG apps with FastAPI", corpus)
    assert evidence_supported("custom answer: require sponsorship => No", corpus)
    assert not evidence_supported("I have no non-compete agreements", corpus)
    assert not evidence_supported("", corpus)
    assert not evidence_supported("planning on relocating to Ontario", corpus)


def test_llm_answers_without_real_evidence_are_rejected():
    html = """
    <label for="a">Are you comfortable with on-call rotations?</label><input id="a">
    <label for="b">Do you know Kubernetes?</label><input id="b">
    <label for="c">Describe your favourite project</label><textarea id="c"></textarea>"""

    class Liar:
        async def call_tool(self, system, user, *rest):
            fields = {f["label"]: f["id"] for f in json.loads(user.split("<form_fields>")[1].split("</form_fields>")[0])}
            return {"answers": [
                # confident, but invents a fact that is nowhere in the profile
                {"id": fields["Are you comfortable with on-call rotations?"], "action": "fill", "value": "Yes",
                 "confidence": 0.95, "evidence": "I have done on-call for two years"},
                # no evidence at all
                {"id": fields["Do you know Kubernetes?"], "action": "fill", "value": "Yes", "confidence": 0.9,
                 "evidence": ""},
                # real evidence from the profile
                {"id": fields["Describe your favourite project"], "action": "fill",
                 "value": "I enjoy building AI features as an AI Engineer.", "confidence": 0.8,
                 "evidence": "current title: AI Engineer"}]}

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(html)
            plan = {a.label: a for a in await plan_fields(await extract_fields(page), PROFILE, llm=Liar())}
            for q in ("Are you comfortable with on-call rotations?", "Do you know Kubernetes?"):
                assert plan[q].action == "skip" and plan[q].needs_review, q
                assert "could not quote" in plan[q].reason, plan[q].reason
            ok = plan["Describe your favourite project"]
            assert ok.action == "fill" and ok.needs_review and "AI Engineer" in ok.reason
            await browser.close()
    run(go())


def test_legal_and_eligibility_questions_never_reach_the_llm():
    from app.planner import is_sensitive
    for q in ("Are you subject to any non-compete, non-solicitation, or confidentiality agreements?",
              "Do you have a legal right to work in Canada if hired by Greenhouse?",
              "Please confirm you're based in BC or Ontario.", "Will you require visa sponsorship?",
              "Are you willing to relocate?", "Acknowledge/Confirm"):
        assert is_sensitive(q), q
    for q in ("Do you have experience with Python?", "Why do you want to work here?", "Github profile"):
        assert not is_sensitive(q), q


# ------------------------------------------------------------------ regressions from a real Workday "My Information" page

WORKDAY_HTML = """
<label for="g">Given Name(s)*</label><input id="g">
<label for="f">Family Name*</label><input id="f">
<label for="lg">Local Given Name(s)</label><input id="lg">
<label for="lf">Local Family Name</label><input id="lf">
<label for="a1">Address Line 1*</label><input id="a1">
<label for="c">City*</label><input id="c">
<label for="pc">Postal Code*</label><input id="pc">
<label for="em">Email address*</label><input id="em" type="email">
<label for="pn">Phone number*</label><input id="pn" type="tel">
<label for="pe">Phone Extension</label><input id="pe">
<label><input type="checkbox" id="tc"> Yes, please also add me to the Talent Community, where my personal
 information is used to send me personalized communications about other job opportunities</label>
<input type="file" id="cv" data-automation-id="file-upload-input-ref">
<input type="file" id="cl" aria-label="Cover letter">
"""


class SkipAll:
    def __init__(self):
        self.seen = []

    async def call_tool(self, system, user, *rest):
        fields = json.loads(user.split("<form_fields>")[1].split("</form_fields>")[0])
        self.seen += [f["label"] for f in fields]
        return {"answers": [{"id": f["id"], "action": "skip", "value": "", "confidence": 0, "evidence": "",
                             "reason": "not stated"} for f in fields]}


def test_workday_style_form_gets_the_right_values_in_the_right_places():
    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(WORKDAY_HTML)
            fields = await extract_fields(page)
            llm = SkipAll()
            plan = {a.label: a for a in await plan_fields(fields, PROFILE, resume_path=Path("resume.pdf"), llm=llm)}

            assert plan["Given Name(s)"].value == "Asha" and plan["Family Name"].value == "Rao"
            assert plan["Email address"].value == "asha@example.com" and plan["Phone number"].value == PROFILE["phone"]
            # the bugs: the phone number landed in "Extension", English names in the "Local" (native script) fields
            assert plan["Phone Extension"].action == "skip" and plan["Phone Extension"].value is None
            assert plan["Local Given Name(s)"].action == "skip" and plan["Local Family Name"].action == "skip"

            # street/postal facts are not in the profile: left blank, and never sent to the LLM to "quote" a city
            for lab in ("Address Line 1", "Postal Code"):
                assert plan[lab].action == "skip" and "not in your profile" in plan[lab].reason, plan[lab].reason
                assert lab not in llm.seen
            # a marketing opt-in is consent: never filled, never sent to the LLM
            optin = [a for a in plan.values() if "Talent Community" in a.label][0]
            assert optin.action == "skip" and optin.needs_review
            assert not any("Talent Community" in s for s in llm.seen)

            # one resume upload, into the unlabelled box; the cover-letter box is left alone
            uploads = [a.label for a in plan.values() if a.action == "upload"]
            assert uploads == ["File upload (file-upload-input-ref)"], uploads

            # with the facts in the profile they are filled from it
            full = {**PROFILE, "address": "12 MG Road", "postal_code": "411001"}
            plan2 = {a.label: a for a in await plan_fields(fields, full, resume_path=Path("resume.pdf"), llm=SkipAll())}
            assert plan2["Address Line 1"].value == "12 MG Road" and plan2["Address Line 1"].source == "profile"
            assert plan2["Postal Code"].value == "411001"
            assert plan2["Email address"].value == "asha@example.com", "an address must never land in an email field"
            await browser.close()
    run(go())


def test_bad_resume_autofill_does_not_preserve_a_phone_number_as_family_name():
    from app.pipeline import fill_application

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content('<label for="family">Family Name</label>'
                                  '<input id="family" value="0000000000">')
            result = await fill_application(page, PROFILE, None, llm=None)
            assert await page.input_value("#family") == PROFILE["last_name"]
            assert result["rows"][0]["status"] == "filled"
            await browser.close()
    run(go())


def test_snapshot_reports_nonempty_but_invalid_required_email():
    from app.pipeline import snapshot

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content('<label for="email">Email address</label>'
                                  '<input id="email" type="email" required value="not-an-email">')
            snap = await snapshot(page)
            assert snap["required_empty"] == ["Email address"]
            assert snap["rows"][0]["value"] == "not-an-email"
            assert snap["rows"][0]["invalid"] is True
            await browser.close()
    run(go())


def test_skip_reason_is_not_overwritten_and_lists_the_real_choices():
    """Real bug: 'skipped - evidence: Company career page' hid the real reason (not one of the options)."""
    html = ('<label for="h">How large a team do you prefer?</label><select id="h"><option value="">Select</option>'
            '<option>Small</option><option>Medium</option><option>Large</option></select>')
    profile = {**PROFILE, "custom_answers": [{"question_pattern": "team", "answer": "Huge"}]}

    class Picky:
        async def call_tool(self, system, user, *rest):
            f = json.loads(user.split("<form_fields>")[1].split("</form_fields>")[0])[0]
            return {"answers": [{"id": f["id"], "action": "select", "value": "Huge",
                                 "confidence": 0.9, "evidence": "custom answer: team => Huge"}]}

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(html)
            a = (await plan_fields(await extract_fields(page), profile, llm=Picky()))[0]
            assert a.action == "skip" and "not one of the available options" in a.reason, a.reason
            assert "choices: Small, Medium, Large" in a.reason, a.reason
            assert "evidence:" not in a.reason
            await browser.close()
    run(go())


# ------------------------------------------------------------------ values wiped by the site after they were filled

WIPE_ONCE = """
<label for="a">First Name</label><input id="a">
<label for="b">Family Name</label><input id="b">
<script>  /* the framework re-renders the first field when the second one changes */
 const a = document.getElementById('a'), b = document.getElementById('b'); let wiped = false;
 b.addEventListener('input', () => { if (!wiped) { wiped = true; setTimeout(() => { a.value = ''; }, 150); } });
</script>"""

WIPE_ALWAYS = """
<label for="a">First Name</label><input id="a">
<label for="b">Family Name</label><input id="b">
<script>
 const a = document.getElementById('a');
 a.addEventListener('input', () => setTimeout(() => { a.value = ''; }, 100));
</script>"""


def test_values_wiped_by_the_site_are_refilled():
    from app.pipeline import fill_application

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(WIPE_ONCE)
            result = await fill_application(page, PROFILE, None, llm=SkipAll())
            rows = {r["label"]: r for r in result["rows"]}
            assert await page.input_value("#a") == "Asha", "the vanished value must be typed again"
            assert rows["First Name"]["status"] == "filled" and "re-filled" in rows["First Name"]["detail"]
            assert result["required_unfilled"] == []
            await browser.close()
    run(go())


def test_values_the_site_keeps_clearing_are_reported_honestly():
    from app.pipeline import fill_application

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(WIPE_ALWAYS)
            result = await fill_application(page, PROFILE, None, llm=SkipAll())
            row = {r["label"]: r for r in result["rows"]}["First Name"]
            # caught by the filler's own settle check ("did not stick") or by the later repair pass ("keeps clearing")
            assert row["status"] == "verified_failed" and row["needs_review"]
            assert "keeps clearing" in row["detail"] or "did not stick" in row["detail"]
            await browser.close()
    run(go())


# ------------------------------------------------------------------ regressions from the Mastercard apply form (run 2)

class Eager:
    """An over-eager model: answers every field it is shown, with 'evidence' that really exists in the profile."""

    def __init__(self, value="Example", evidence="first name: Asha"):
        self.seen, self.value, self.evidence = [], value, evidence

    async def call_tool(self, system, user, *rest):
        fields = json.loads(user.split("<form_fields>")[1].split("</form_fields>")[0])
        self.seen += [f["label"] for f in fields]
        return {"answers": [{"id": f["id"], "action": "fill", "value": self.value, "confidence": 0.9,
                             "evidence": self.evidence} for f in fields]}


def test_personal_facts_are_never_guessed_by_the_model():
    html = """
    <label for="a">Local Given Name(s)</label><input id="a">
    <label for="b">Local Family Name</label><input id="b">
    <label for="c">How Did You Hear About Us?</label><input id="c">
    <label for="d">Have you ever worked for Mastercard as an employee or provided services as a contingent worker?</label><input id="d">
    <label for="e">Source</label><input id="e">
    <label for="f">Describe your open source contributions</label><textarea id="f"></textarea>"""

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(html)
            fields = await extract_fields(page)
            llm = Eager()
            plan = {a.label[:24]: a for a in await plan_fields(fields, PROFILE, llm=llm)}
            # none of the personal ones reached the model; the essay question (open SOURCE) still does
            assert llm.seen == ["Describe your open source contributions"], llm.seen
            for key in ("Local Given Name(s)", "Local Family Name", "How Did You Hear About Us?", "Have you ever worked for", "Source"):
                assert plan[key[:24]].action == "skip" and "personal fact" in plan[key[:24]].reason, (key, plan[key[:24]].reason)

            # YOU can state them: profile keys, or a custom answer for your own history
            mine = {**PROFILE, "local_first_name": "Local-A", "local_last_name": "Local-B", "referral_source": "LinkedIn",
                    "custom_answers": [{"question_pattern": "ever worked for", "answer": "No"}]}
            plan2 = {a.label[:24]: a for a in await plan_fields(fields, mine, llm=Eager())}
            assert plan2["Local Given Name(s)"].value == "Local-A" and plan2["Local Family Name"].value == "Local-B"
            hear = plan2["How Did You Hear About Us?"[:24]]
            assert hear.value == "LinkedIn" and hear.source == "profile"
            assert plan2["Have you ever worked for"].value == "No" and plan2["Have you ever worked for"].source == "custom"
            await browser.close()
    run(go())


def test_referral_source_must_match_a_real_option_else_it_is_left_to_you():
    html = ('<label for="h">How Did You Hear About Us?</label><select id="h"><option value="">Select</option>'
            '<option>CareerArc</option><option>LinkedIn</option><option>Referral</option></select>')

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(html)
            fields = await extract_fields(page)
            llm = Eager(value="CareerArc", evidence="Company career page")
            # "Company career page" is not an option; "CareerArc" is NOT "career page". Nothing may be picked.
            a = (await plan_fields(fields, {**PROFILE, "referral_source": "Company career page"}, llm=llm))[0]
            assert a.action == "skip" and llm.seen == [], (a.action, a.reason, llm.seen)
            ok = (await plan_fields(fields, {**PROFILE, "referral_source": "LinkedIn"}, llm=llm))[0]
            assert ok.action == "select" and ok.value == "LinkedIn" and ok.source == "profile"
            await browser.close()
    run(go())


def test_a_dropdown_choice_must_be_stated_in_the_quoted_evidence():
    """Real bug: the model picked 'CareerArc' and quoted 'Company career page' as its evidence."""
    html = ('<label for="w">Preferred work arrangement</label><select id="w"><option value="">Select</option>'
            '<option>Remote</option><option>Hybrid</option><option>On-site</option></select>')
    profile = {**PROFILE, "work_style": "Remote"}

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(html)
            fields = await extract_fields(page)
            wrong = (await plan_fields(fields, profile, llm=Eager(value="Hybrid", evidence="work style: Remote")))[0]
            assert wrong.action == "skip" and "'Hybrid'" in wrong.reason and "Choose it yourself" in wrong.reason
            right = (await plan_fields(fields, profile, llm=Eager(value="Remote", evidence="work style: Remote")))[0]
            assert right.action == "select" and right.value == "Remote" and right.needs_review
            await browser.close()
    run(go())


def test_phone_device_type_comes_from_the_profile_not_a_guess():
    html = ('<label for="t">Phone Device Type</label><select id="t"><option value="">Select</option>'
            '<option>Mobile</option><option>Home</option><option>Work</option></select>')

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(html)
            fields = await extract_fields(page)
            a = (await plan_fields(fields, {**PROFILE, "phone_type": "Mobile"}, llm=None))[0]
            assert a.action == "select" and a.value == "Mobile" and a.source == "profile"
            # without the key a model may only suggest it if its quote actually says so
            b = (await plan_fields(fields, PROFILE, llm=Eager(value="Mobile", evidence="phone: +91 90000 00000")))[0]
            assert b.action == "skip" and "Choose it yourself" in b.reason
            await browser.close()
    run(go())


GROWING_FORM = """
<label for="h">Preferred contact method</label>
<select id="h" onchange="if (this.value === 'Other') document.getElementById('more').style.display = 'block'">
  <option value="">Select</option><option>Email</option><option>Other</option></select>
<div id="more" style="display:none"><label for="sp">Please specify</label><input id="sp"></div>"""


def test_fields_that_appear_after_an_answer_are_filled_too():
    """Real case: a dependent 'Source' dropdown appeared after the first answer and was never handled."""
    from app.pipeline import fill_application
    profile = {**PROFILE, "custom_answers": [{"question_pattern": "contact method", "answer": "Other"},
                                             {"question_pattern": "please specify", "answer": "Signal"}]}

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(GROWING_FORM)
            result = await fill_application(page, profile, None, llm=None)
            assert await page.input_value("#h") == "Other"
            assert await page.input_value("#sp") == "Signal", "the field that appeared later must be filled"
            labels = [r["label"] for r in result["rows"]]
            assert labels == ["Preferred contact method", "Please specify"], labels
            assert result["filled"] == 2 and result["required_unfilled"] == []
            await browser.close()
    run(go())


def test_fill_next_steps_advances_and_fills_until_no_next_button_remains():
    from app.pipeline import fill_application
    from app.sessions import Command, Session, SessionManager

    html = """
    <section id="step-one">
      <label for="first">First Name</label><input id="first">
      <button type="button" id="next">Next</button>
    </section>
    <section id="step-two" hidden>
      <label for="city">Current City</label><input id="city">
      <button type="button" id="submit">Submit application</button>
    </section>
    <script>
      document.getElementById('next').addEventListener('click', () => {
        document.getElementById('step-one').hidden = true;
        document.getElementById('step-two').hidden = false;
      });
      document.getElementById('submit').addEventListener('click', () => { window.submitted = true; });
    </script>
    """

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(html)
            s = Session(id=1, url="http://example.test/app", page=page)
            s.refill = lambda: fill_application(page, PROFILE, None, llm=None)
            s.apply_result(await s.refill())
            manager = SessionManager(lambda _: asyncio.sleep(0))
            result = await manager._advance_steps(s)
            assert result["advanced"] == 1, result
            assert await page.input_value("#city") == "Pune"
            assert {row["label"] for row in s.rows} == {"First Name", "Current City"}
            assert await page.evaluate("() => window.submitted !== true"), "navigation must stop before Submit"
            await browser.close()
    run(go())


# ------------------------------------------------------------------ regressions from a Greenhouse run (phone widget, noisy highlights)

MASKED_PHONE = """
<label for="ph">Phone*</label><input id="ph" type="tel" required>
<script>  /* like intl-tel-input: reformats while you type, so the stored text differs from what was typed */
 const ph = document.getElementById('ph');
 ph.addEventListener('input', () => { const d = ph.value.replace(/\\D/g, ''); ph.value = d.length > 5 ? d.slice(0, 5) + ' ' + d.slice(5) : d; });
</script>"""


def test_a_reformatted_phone_number_is_not_reported_as_a_failure():
    """Real bug: the widget turned the typed number into '73502 68634', which the exact comparison rejected."""
    from app.pipeline import fill_application

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(MASKED_PHONE)
            result = await fill_application(page, PROFILE, None, llm=None)
            row = result["rows"][0]
            assert await page.input_value("#ph") != PROFILE["phone"], "the widget really did reformat the value"
            assert row["status"] == "filled", row
            assert result["required_unfilled"] == []
            await browser.close()
    run(go())


def test_same_text_ignores_formatting_but_not_content():
    from app.filler import _same_text
    assert _same_text("00000 00000", "0000000000") and _same_text("+91 00000 00000", "0000000000")
    assert _same_text("411 001", "411001") and _same_text("Pune", "pune") and _same_text("आदित्य", "आदित्य")
    assert not _same_text("00000 00001", "0000000000") and not _same_text("", "Pune") and not _same_text("Pune", "Mumbai")


HIGHLIGHT_FORM = """
<label for="a">Nickname</label><input id="a">
<label for="b">Favourite colour*</label><input id="b" required>
<label><input type="checkbox" id="c"> I agree to receive newsletters</label>
<label><input type="checkbox" id="d" required> I acknowledge the privacy notice*</label>
<label for="e">Describe your favourite project</label><textarea id="e"></textarea>
<label for="f">If you chose Other, please specify here:</label><input id="f">"""


def test_only_things_that_need_the_user_are_highlighted():
    """An optional blank (a nickname, a newsletter opt-in) is not a warning; a required blank is."""
    from app.pipeline import fill_application

    class OnlyProject:
        def __init__(self):
            self.seen = []

        async def call_tool(self, system, user, *rest):
            fields = json.loads(user.split("<form_fields>")[1].split("</form_fields>")[0])
            self.seen += [f["label"] for f in fields]
            out = []
            for f in fields:
                if "project" in f["label"].lower():
                    out.append({"id": f["id"], "action": "fill", "value": "Built LLM apps as an AI Engineer.",
                                "confidence": 0.8, "evidence": "current title: AI Engineer"})
                else:
                    out.append({"id": f["id"], "action": "skip", "value": "", "confidence": 0, "evidence": "",
                                "reason": "not stated"})
            return {"answers": out}

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(HIGHLIGHT_FORM)
            llm = OnlyProject()
            result = await fill_application(page, PROFILE, None, llm=llm)
            rows = {r["label"]: r for r in result["rows"]}
            assert rows["Nickname"]["needs_review"] is False and rows["Nickname"]["required"] is False
            assert rows["I agree to receive newsletters"]["needs_review"] is False, "an optional opt-in left blank is fine"
            assert rows["Favourite colour"]["needs_review"] and rows["Favourite colour"]["required"]
            assert rows["I acknowledge the privacy notice"]["needs_review"], "a required consent box needs the user"
            assert rows["Describe your favourite project"]["needs_review"], "AI-written answers are always highlighted"
            assert result["needs_review"] == 3, [(r["label"], r["needs_review"]) for r in result["rows"]]
            assert "If you chose Other, please specify here:" not in llm.seen, "follow-up boxes never go to the model"
            await browser.close()
    run(go())

# ------------------------------------------------------------------ regressions from a Mastercard (Phenom) run: multi-page walk

def test_radio_inside_a_whole_form_fieldset_gets_its_own_question_and_required_flag():
    """Real bug: the form-wide legend 'You are applying for -' became the radio's label, and the
    aria-required on the radiogroup was missed, so a required question was skipped silently."""
    html = """
    <form><fieldset><legend>You are applying for - Principal Data Engineer</legend>
      <label for="fn">Given Name(s)</label><input id="fn">
      <div role="radiogroup" aria-labelledby="pw-q" aria-required="true">
        <label id="pw-q" for="pw">Have you ever worked for Mastercard?<span aria-hidden="true">*</span></label>
        <div id="pw" role="radiogroup" aria-label="Have you ever worked for Mastercard?">
          <label><input type="radio" name="previousworker" value="Yes" aria-label="Yes"> Yes</label>
          <label><input type="radio" name="previousworker" value="No" aria-label="No"> No</label>
        </div></div>
      <fieldset><legend>Preferred shift</legend>
        <label><input type="radio" name="shift" value="d"> Day</label>
        <label><input type="radio" name="shift" value="n"> Night</label></fieldset>
    </fieldset></form>"""

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(html)
            radios = {f.name: f for f in await extract_fields(page) if f.type == "radio"}
            assert radios["previousworker"].label == "Have you ever worked for Mastercard?"
            assert radios["previousworker"].required is True
            assert radios["shift"].label == "Preferred shift", "a tight fieldset still names its own group"
            await browser.close()
    run(go())


def test_snapshot_reports_a_filled_field_the_site_itself_calls_invalid():
    from app.pipeline import snapshot

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(
                '<div class="field has-error"><label for="email">Email address</label>'
                '<input id="email" type="email" value="a@b.co" aria-invalid="true">'
                '<span id="email-errorMsg" class="error">Please enter email address.</span></div>'
                '<div class="field"><label for="city">City</label><input id="city" value="Pune"></div>')
            snap = await snapshot(page)
            assert snap["required_empty"] == ["Email address"]
            assert snap["rows"][0]["error"] == "Please enter email address."
            assert snap["rows"][1]["invalid"] is False, "an error in another row must not leak into this one"
            await browser.close()
    run(go())


# Three pages + the site's own Review page. Page 2's email box behaves like a React form whose state is
# wiped when another field changes: the text stays visible but the site rejects it until it is typed again.
MULTI_STEP = r"""
<ol class="steps"><li aria-current="step" id="crumb">My Information</li></ol>
<a href="#" id="crumb-review">Review</a>
<form onsubmit="return false">
<section data-step="1">
  <h2>My Information</h2>
  <label for="first">First Name *</label><input id="first" required>
  <label for="last">Last Name</label><input id="last">
</section>
<section data-step="2" hidden>
  <h2>Contact</h2>
  <div class="row"><label for="email">Email address *</label><input id="email" type="email" required>
    <span id="email-errorMsg" class="error" hidden>Please enter email address.</span></div>
  <div class="row"><label for="phone">Phone</label><input id="phone" type="tel"></div>
</section>
<section data-step="3" hidden>
  <h2>Location</h2>
  <label for="city">Current City</label><input id="city">
</section>
<section data-step="4" hidden>
  <h2>Review</h2><div id="summary"></div>
</section>
<button type="button" id="next">Next</button>
<button type="button" id="submit" hidden>Submit</button>
</form>
<script>
  let step = 1, emailState = '';
  const $ = id => document.getElementById(id);
  const names = {1: 'My Information', 2: 'Contact', 3: 'Location', 4: 'Review'};
  $('email').addEventListener('keydown', () => { setTimeout(() => { emailState = $('email').value; }, 0); });
  $('phone').addEventListener('input', () => { emailState = ''; });  // a re-render wipes the email's state
  const show = n => {
    document.querySelectorAll('section').forEach(s => s.hidden = s.dataset.step != n);
    $('crumb').textContent = names[n];
    $('next').hidden = n === 4; $('submit').hidden = n !== 4;
    if (n === 4) $('summary').innerText = ['First Name: ' + $('first').value, 'Email: ' + $('email').value,
                                           'City: ' + $('city').value].join('\n');
  };
  $('next').addEventListener('click', () => {
    if (step === 1 && !$('first').value) return;
    if (step === 2 && !emailState) { $('email-errorMsg').hidden = false; $('email').setAttribute('aria-invalid', 'true'); return; }
    $('email-errorMsg').hidden = true; $('email').removeAttribute('aria-invalid');
    step += 1; show(step);
  });
  $('submit').addEventListener('click', () => { window.submitted = true; });
</script>
"""


def _walk_session(page, profile):
    from app.pipeline import fill_application
    from app.sessions import Session, SessionManager

    s = Session(id=1, url="http://example.test/app", page=page)
    s.refill = lambda: fill_application(page, profile, None, llm=None)
    return s, SessionManager(lambda _: asyncio.sleep(0))


def test_walks_every_page_repairs_rejected_values_and_stops_on_the_final_review_page():
    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(MULTI_STEP)
            s, manager = _walk_session(page, PROFILE)
            s.step_name = await manager._step_name(page)
            s.apply_result(await s.refill())
            out = await manager._advance_steps(s)
            assert out["why"] == "final" and out["status"] == "ready_to_submit", out
            assert out["advanced"] == 3 and s.step == 4 and s.final_reached
            assert await page.evaluate("() => window.submitted !== true"), "Submit is only for the user to approve"
            by_label = {r["label"]: r for r in s.rows}
            assert by_label["First Name"]["step"] == 1 and by_label["First Name"]["step_name"] == "My Information"
            assert by_label["Email address"]["step"] == 2 and by_label["Current City"]["step"] == 3
            assert "Email: asha@example.com" in s.review_text and "City: Pune" in s.review_text
            await browser.close()
    run(go())


def test_a_page_the_site_refuses_is_reported_with_the_sites_own_message():
    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            # this site never accepts the email, however it is typed
            await page.set_content(MULTI_STEP.replace("emailState = $('email').value;", ""))
            s, manager = _walk_session(page, PROFILE)
            s.apply_result(await s.refill())
            out = await manager._advance_steps(s)
            assert out["why"] == "stalled" and out["advanced"] == 1, out
            assert s.blockers == ["Email address: Please enter email address."], s.blockers
            assert "Please enter email address." in out["message"]
            await browser.close()
    run(go())
