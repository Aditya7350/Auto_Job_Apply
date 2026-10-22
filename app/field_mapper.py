import re
from dataclasses import dataclass
from typing import Optional


# Brand names written in camelCase ("GitHub", "LinkedIn") must not be split into "git hub".
BRANDS = ("github", "gitlab", "linkedin", "leetcode", "youtube", "stackoverflow", "hackerrank", "codechef")


def norm(value: str) -> str:
    value = value or ""
    for brand in BRANDS:
        value = re.sub(brand, brand, value, flags=re.I)
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


@dataclass
class CandidateField:
    key: str
    value: str
    aliases: tuple[str, ...]
    negative: tuple[str, ...] = ()


# Ordered from specific -> generic. This prevents "name" from stealing
# fields such as first_name or company_name.
FIELD_DEFINITIONS = [
    CandidateField("first_name", "", ("first name", "given name", "forename", "fname"), ("local", "native")),
    CandidateField("last_name", "", ("last name", "surname", "family name", "lname"), ("local", "native")),
    CandidateField("full_name", "", ("full name", "candidate name", "applicant name", "your name"), ("company name", "local", "native")),
    CandidateField("email", "", ("email address", "email", "e mail", "mail")),
    CandidateField("phone", "", ("phone number", "mobile number", "mobile", "telephone number", "phone", "contact number"),
                   ("extension", "ext", "device type", "phone type", "phone code", "country code", "area code", "dial code")),
    CandidateField("linkedin", "", ("linkedin profile", "linkedin url", "linkedin", "linkedin profile url")),
    CandidateField("github", "", ("github profile", "github url", "github", "github profile url")),
    CandidateField("portfolio", "", ("portfolio url", "portfolio", "personal website", "website url", "personal website url")),
    CandidateField("city", "", ("current city", "city", "town")),
    CandidateField("state", "", ("state", "province", "region")),
    CandidateField("country", "", ("country", "country of residence", "current country")),
    CandidateField("address", "", ("address", "address line 1", "address line one", "street address", "current address"),
                   ("email", "e mail", "web", "website", "ip")),
    CandidateField("local_first_name", "", ("local given name", "local given names", "local first name", "native first name")),
    CandidateField("local_last_name", "", ("local family name", "local last name", "native last name")),
    CandidateField("phone_type", "", ("phone device type", "phone type", "device type", "type of phone")),
    CandidateField("referral_source", "", ("how did you hear about us", "how did you hear about this job",
                                           "how did you hear about this position", "how did you hear",
                                           "where did you hear about us", "referral source", "hear about",
                                           "how did you find out", "how did you find", "how did you learn about",
                                           "where did you learn about", "where did you find", "how did you come across",
                                           "application source", "how you heard")),
    CandidateField("postal_code", "", ("postal code", "zip code", "zip", "pin code", "pincode", "postcode")),
    CandidateField("years_experience", "", ("years of experience", "years experience", "total experience", "experience in years")),
    CandidateField("current_title", "", ("current job title", "current title", "current role", "job title"), ("job title you are applying for",)),
    CandidateField("notice_period", "", ("notice period", "availability", "when can you start")),
]


def build_definitions(profile: dict):
    definitions = []
    for item in FIELD_DEFINITIONS:
        definitions.append(CandidateField(item.key, str(profile.get(item.key, "") or ""), item.aliases, item.negative))
    return definitions


def score_field(metadata: str, definition: CandidateField) -> int:
    text = norm(metadata)
    if not text or not definition.value:
        return -1
    if any(f" {norm(n)} " in f" {text} " for n in definition.negative):
        return -1

    score = 0
    for alias in definition.aliases:
        a = norm(alias)
        if text == a:
            score = max(score, 100)
        elif f" {a} " in f" {text} ":
            score = max(score, 70 if len(a.split()) > 1 else 45)
        elif any(token in text.split() for token in a.split()):
            score = max(score, 20)

    # Attribute hints are useful when labels are abbreviated.
    compact = text.replace(" ", "")
    for alias in definition.aliases:
        ac = norm(alias).replace(" ", "")
        if compact == ac:
            score = max(score, 90)
    return score


def choose_field(metadata: str, definitions: list[CandidateField]) -> Optional[CandidateField]:
    scored = [(score_field(metadata, d), d) for d in definitions]
    scored = [(s, d) for s, d in scored if s >= 45]
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]