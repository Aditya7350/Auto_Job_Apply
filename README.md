# Free Auto Job Apply V4

V4 adds application-start detection to the free Playwright MVP.

Flow:
1. Open supplied job URL.
2. If the page already contains form controls, fill them directly.
3. Otherwise detect a likely Apply/Application button.
4. Click Apply and support either same-tab navigation or a new tab.
5. Detect login gates and pause for manual login.
6. Continue intelligent field mapping and resume upload after login.
7. Walk every page of a multi-step form (up to 15): fill the page, press Next/Continue, fill the new page, repeat.
   Values the site silently rejected (text visible but "Please enter email address") are re-typed key by key
   and Next is tried once more. Cookie layers that cover the form are dismissed on every page.
8. Stop at the final page, the one whose only way forward is Submit (status **ready to submit**). The review page
   then shows every field from every page (with its page number) and the text of the site's own Review page.
9. If a page cannot be passed, it stops there and lists exactly which fields block it, with the site's own error
   message. Fix them in the browser and press **Fill next steps** to continue.
10. Keep the browser open for human review; Submit is only pressed when you click **Approve & submit**.

Run on Windows:
```powershell
Copy-Item config\profile.example.json config\profile.json
# Edit config\profile.json with your details before running.
.venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
uvicorn app.main:app
```

Keep your profile, `.env`, database, uploaded resume, and logs private; they are excluded from Git.

Open http://127.0.0.1:8000

Do not enter passwords into the automation. CAPTCHA/OTP are not bypassed.
