"""Reproduces the problems seen on a real Greenhouse form (react-select dropdowns).

* every required dropdown showed up twice (react-select's invisible `required` twin input)
* options of a dropdown are unknown until it is opened, so the LLM guessed a value that does not exist
* the chosen value is shown in a separate element, so a filled dropdown looked empty
* React re-renders replaced elements, so the handle tag disappeared ("waiting for locator data-aja-id")
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from playwright.async_api import async_playwright  # noqa: E402

from app.extractor import extract_fields, probe_combobox_options  # noqa: E402
from app.filler import apply_plan  # noqa: E402
from app.pipeline import fill_application, snapshot  # noqa: E402
from app.planner import plan_fields  # noqa: E402
from test_engine import PROFILE  # noqa: E402

TWIN_A = ('<input tabindex="-1" aria-hidden="true" required value="" autocomplete="off" '
          'style="opacity:0;pointer-events:none;position:absolute;bottom:0;left:0;right:0;width:100%">')
TWIN_B = ('<input tabindex="-1" required value="" '
          'style="opacity:0;pointer-events:none;position:absolute;bottom:0;left:0;right:0;width:100%">')

HTML = """
<form>
 <div><label for="fn">First Name*</label><input id="fn" name="first_name" required></div>
 <div id="s1"></div><div id="s2"></div><div id="s3"></div><div id="s4"></div><div id="s5"></div>
 <label for="resume">Attach</label><input type="file" id="resume" name="resume">
 <label for="cover_letter">Attach</label><input type="file" id="cover_letter" name="cover_letter">
</form>
<script>
function makeSelect(host, id, label, options, twin) {
  host.className = 'select__container'; host.style.cssText = 'position:relative;margin:10px 0';
  host.innerHTML = '<label id="'+id+'-label" for="'+id+'">'+label+'*</label>' +
   '<div class="select__control"><div class="select__value-container">' +
   '<div class="select__placeholder">Select...</div>' +
   '<div class="select__input-container"><input id="'+id+'" class="select__input" type="text" role="combobox" ' +
   'aria-expanded="false" aria-autocomplete="list" aria-controls="'+id+'-listbox" autocomplete="off" ' +
   'aria-labelledby="'+id+'-label"></div></div></div><div class="twin"></div>' +
   '<div id="'+id+'-listbox" role="listbox" style="display:none"></div>';
  const input = host.querySelector('input.select__input'), list = host.querySelector('[role=listbox]');
  const putTwin = () => { host.querySelector('.twin').innerHTML = twin; };  // React re-renders it on every change
  putTwin();
  const render = () => {
    const v = input.value.toLowerCase(); list.innerHTML = '';
    options.filter(o => o.toLowerCase().includes(v)).forEach(o => {
      const d = document.createElement('div'); d.setAttribute('role', 'option'); d.textContent = o;
      d.onmousedown = e => { e.preventDefault(); choose(o); }; list.appendChild(d); });
    list.style.display = 'block';
  };
  function choose(o) {
    const ph = host.querySelector('.select__placeholder'); if (ph) ph.remove();
    let sv = host.querySelector('.select__single-value');
    if (!sv) { sv = document.createElement('div'); sv.className = 'select__single-value';
               host.querySelector('.select__value-container').prepend(sv); }
    sv.textContent = o; input.value = ''; list.style.display = 'none'; putTwin();
    // simulate React replacing the DOM node: our data-aja-id tag disappears
    if (window.__rerender) { const c = input.cloneNode(true); c.removeAttribute('data-aja-id'); input.replaceWith(c); }
  }
  input.addEventListener('focus', render); input.addEventListener('click', render); input.addEventListener('input', render);
  input.addEventListener('keydown', e => { if (e.key === 'Escape') list.style.display = 'none'; });
  host.choose = choose;
}
makeSelect(document.getElementById('s1'), 'country', 'Country', ['Canada', 'India', 'United States'], %(A)s);
makeSelect(document.getElementById('s2'), 'hear', 'How did you initially hear about this job?',
           ['LinkedIn', 'Company website', 'Job board', 'Other'], %(B)s);
makeSelect(document.getElementById('s3'), 'canada', 'Do you have a legal right to work in Canada?', ['Yes', 'No'], %(A)s);
makeSelect(document.getElementById('s4'), 'tz', 'Time zone', ['Asia/Kolkata', 'America/Toronto'], %(A)s);
makeSelect(document.getElementById('s5'), 'arr', 'Preferred work arrangement', ['Remote', 'Hybrid', 'On-site'], %(A)s);
document.getElementById('s4').choose('Asia/Kolkata');   // already selected by the user / the site
</script>
""" % {"A": json.dumps(TWIN_A), "B": json.dumps(TWIN_B)}


class FakeLLM:
    def __init__(self):
        self.prompts = []

    async def call_tool(self, system, user, tool_name, tool_description, schema):
        self.prompts.append(user)
        fields = json.loads(user.split("<form_fields>")[1].split("</form_fields>")[0])
        out = []
        for f in fields:
            lab = f["label"].lower()
            if "work arrangement" in lab:
                assert "Hybrid" in f.get("options", []), "the LLM must see the dropdown's real options"
                out.append({"id": f["id"], "action": "select", "value": "Remote", "confidence": 0.9,
                            "evidence": "work style: Remote"})
            else:
                out.append({"id": f["id"], "action": "skip", "value": "", "confidence": 0, "evidence": "",
                            "reason": "profile has no information"})
        return {"answers": out}


def run(coro):
    return asyncio.run(coro)


async def _page(pw, rerender=False):
    browser = await pw.chromium.launch()
    page = await browser.new_page()
    if rerender:
        await page.add_init_script("window.__rerender = true")
    await page.set_content(HTML)
    await page.wait_for_timeout(200)
    return browser, page


def test_no_duplicate_fields_and_options_are_discovered():
    async def go():
        async with async_playwright() as pw:
            browser, page = await _page(pw)
            fields = await extract_fields(page)
            labels = [f.label for f in fields]
            for lab in ("Country", "How did you initially hear about this job?",
                        "Do you have a legal right to work in Canada?", "Time zone", "Preferred work arrangement"):
                assert labels.count(lab) == 1, f"{lab!r} appears {labels.count(lab)}x: {labels}"
            assert sum(1 for f in fields if f.type == "combobox") == 5

            files = [f.label for f in fields if f.type == "file"]
            assert files == ["Attach (resume)", "Attach (cover_letter)"], files

            by = {f.label: f for f in fields}
            assert by["Time zone"].value == "Asia/Kolkata", "selected value must be read from the display element"
            assert by["Country"].value == ""

            await probe_combobox_options(fields)
            assert [o.label for o in by["Country"].options] == ["Canada", "India", "United States"]
            assert by["Time zone"].options == [], "already-selected dropdowns are not probed"
            assert await page.locator('[role=option]:visible').count() == 0, "menus must be closed again"
            await browser.close()
    run(go())


def test_fill_picks_real_options_and_reports_unanswerable_questions():
    async def go():
        async with async_playwright() as pw:
            browser, page = await _page(pw)
            llm = FakeLLM()
            # how you found the job is YOUR fact: it comes from the profile, never from the model
            profile = {**PROFILE, "referral_source": "Company website", "work_style": "Remote"}
            result = await fill_application(page, profile, None, llm=llm)
            rows = {r["label"]: r for r in result["rows"]}

            assert rows["Country"]["status"] == "filled" and rows["Country"]["value"] == "India"
            assert rows["How did you initially hear about this job?"]["status"] == "filled"
            assert rows["How did you initially hear about this job?"]["value"] == "Company website"
            assert rows["How did you initially hear about this job?"]["source"] == "profile"
            assert rows["Preferred work arrangement"]["value"] == "Remote" and rows["Preferred work arrangement"]["source"] == "llm"
            assert all("hear about" not in p for p in llm.prompts), "referral questions must never reach the LLM"
            assert rows["Time zone"]["status"] == "kept", "an existing selection must never be overwritten"
            canada = rows["Do you have a legal right to work in Canada?"]
            assert canada["needs_review"] and "answer this yourself" in canada["detail"]
            assert all("legal right" not in p for p in llm.prompts), "legal questions must not be sent to the LLM"

            texts = {i: await page.locator(f"#{i}").locator("xpath=ancestor::div[contains(@class,'select__container')]"
                                                              "//div[contains(@class,'select__single-value')]").all_inner_texts()
                     for i in ("country", "hear", "canada", "tz", "arr")}
            assert texts == {"country": ["India"], "hear": ["Company website"], "canada": [], "tz": ["Asia/Kolkata"],
                             "arr": ["Remote"]}, texts

            snap = await snapshot(page)
            assert snap["required_empty"] == ["Do you have a legal right to work in Canada?"], snap["required_empty"]
            assert result["required_unfilled"] == ["Do you have a legal right to work in Canada?"]
            await browser.close()
    run(go())


def test_recovers_when_the_site_replaces_elements():
    """React swaps DOM nodes after a change; the handle tag vanishes. The filler must re-find fields."""
    async def go():
        async with async_playwright() as pw:
            browser, page = await _page(pw, rerender=True)
            fields = await extract_fields(page)
            await probe_combobox_options(fields)
            plan = await plan_fields(fields, PROFILE, llm=FakeLLM())
            # worst case: every tag is gone before we start filling
            await page.evaluate("document.querySelectorAll('[data-aja-id]').forEach(e => e.removeAttribute('data-aja-id'))")
            results = await apply_plan(page, fields, plan)
            by = {r.label: r for r in results}
            assert by["Country"].status == "filled", by["Country"]
            assert by["First Name"].status == "filled", by["First Name"]
            assert await page.input_value("#fn") == "Asha"
            await browser.close()
    run(go())


TYPEAHEAD_HTML = """
<div style="position:relative">
 <label for="city">Location (City)*</label>
 <input id="city" role="combobox" aria-autocomplete="list" aria-controls="city-list" autocomplete="off">
 <div class="shown"></div>
 <div id="city-list" role="listbox"></div>
</div>
<script>
 const input = document.getElementById('city'), list = document.getElementById('city-list');
 let timer;
 input.addEventListener('input', () => {            // suggestions arrive 1.5 s later, like a places API
   clearTimeout(timer); list.innerHTML = '';
   if (!input.value) return;
   timer = setTimeout(() => {
     ['Pune, Maharashtra, India', 'Punta Gorda, FL, USA'].forEach(t => {
       const d = document.createElement('div'); d.setAttribute('role', 'option'); d.textContent = t;
       d.onmousedown = e => { e.preventDefault(); const s = document.querySelector('.shown');
                              s.className = 'select__single-value'; s.textContent = t; input.value = ''; list.innerHTML = ''; };
       list.appendChild(d); });
   }, 1500);
 });
</script>
"""


def test_type_ahead_city_waits_for_slow_suggestions():
    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(TYPEAHEAD_HTML)
            fields = await extract_fields(page)
            assert [f.type for f in fields] == ["combobox"] and fields[0].options == []
            plan = await plan_fields(fields, PROFILE, llm=None)
            assert plan[0].source == "profile" and plan[0].value == "Pune"
            results = await apply_plan(page, fields, plan)
            assert results[0].status == "filled", results[0]
            assert await page.locator(".select__single-value").inner_text() == "Pune, Maharashtra, India"
            await browser.close()
    run(go())


def test_sensitive_and_consent_fields_are_never_sent_to_the_llm():
    html = """
    <label for="a">Do you consent to interview recording?</label><input id="a">
    <label for="b">Please acknowledge the privacy notice</label><input id="b">
    <label for="c">Pay range comfortable?</label><input id="c">
    <label for="d">What is your favourite language?</label><input id="d">"""

    class Greedy:
        def __init__(self):
            self.seen = []

        async def call_tool(self, system, user, *a):
            fields = json.loads(user.split("<form_fields>")[1].split("</form_fields>")[0])
            self.seen += [f["label"] for f in fields]
            return {"answers": [{"id": f["id"], "action": "fill", "value": "Yes", "confidence": 1.0} for f in fields]}

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(html)
            fields = await extract_fields(page)
            llm = Greedy()
            plan = {a.label: a for a in await plan_fields(fields, PROFILE, llm=llm)}
            assert llm.seen == ["What is your favourite language?"], llm.seen
            for label in ("Do you consent to interview recording?", "Please acknowledge the privacy notice", "Pay range comfortable?"):
                assert plan[label].action == "skip" and plan[label].needs_review, label
            assert plan["What is your favourite language?"].needs_review, "AI answers are always highlighted"
            await browser.close()
    run(go())


PHONE_HTML = """
<div style="position:relative">
 <label for="cc">Country*</label>
 <input id="cc" role="combobox" aria-autocomplete="list" aria-controls="cc-list" autocomplete="off">
 <div class="shown"></div>
 <div id="cc-list" role="listbox" style="display:none"></div>
</div>
<script>
 const OPT = ['Afghanistan +93', 'British Indian Ocean Territory +246', 'India +91', 'Indonesia +62', 'Iran +98'];
 const input = document.getElementById('cc'), list = document.getElementById('cc-list');
 const render = () => { const v = input.value.toLowerCase(); list.innerHTML = '';
   OPT.filter(o => o.toLowerCase().includes(v)).forEach(o => {
     const d = document.createElement('div'); d.setAttribute('role', 'option'); d.textContent = o;
     d.onmousedown = e => { e.preventDefault(); const s = document.querySelector('.shown');
       s.className = 'select__single-value'; s.textContent = o; input.value = ''; list.style.display = 'none'; };
     list.appendChild(d); }); list.style.display = 'block'; };
 input.addEventListener('focus', render); input.addEventListener('click', render); input.addEventListener('input', render);
 input.addEventListener('keydown', e => { if (e.key === 'Escape') list.style.display = 'none'; });
</script>
"""


def test_phone_country_code_picks_india_not_british_indian_ocean_territory():
    """Real bug from a Greenhouse run: the phone's country dropdown ended up as '+246'."""
    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(PHONE_HTML)
            result = await fill_application(page, PROFILE, None, llm=None)
            row = result["rows"][0]
            assert row["status"] == "filled" and row["value"] == "India +91", row
            assert await page.locator(".select__single-value").inner_text() == "India +91"
            await browser.close()
    run(go())