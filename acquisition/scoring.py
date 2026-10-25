"""
Lead scoring: rank leads by how likely they are to become a high-ticket SIP client.

Score (0–100) =
    capacity      up to 40  — estimated monthly SIP potential vs. ₹50K target
    profession    up to 15
    source        up to 20  — referrals convert best
    life stage    up to 10  — 28–50 is the prime SIP-starting window
    existing inv. up to 10  — already invests = less education needed
    consent bonus 5
"""

from dataclasses import dataclass

from . import config


@dataclass
class ScoreResult:
    score: int
    tier: str
    sip_potential: int
    reasons: list[str]


def estimate_sip_potential(monthly_income: int) -> int:
    for floor, ratio in config.SURPLUS_RATIO_BY_INCOME:
        if monthly_income >= floor:
            return int(round(monthly_income * ratio, -2))
    return 0


def score_lead(lead: dict) -> ScoreResult:
    reasons = []
    income = int(lead.get("monthly_income") or 0)
    potential = estimate_sip_potential(income)

    capacity = min(40, int(40 * potential / config.TARGET_SIP_PER_CLIENT))
    reasons.append(f"SIP potential ≈ ₹{potential:,}/mo (+{capacity})")

    profession = (lead.get("profession") or "other").lower()
    prof_pts = config.PROFESSION_POINTS.get(profession, config.PROFESSION_POINTS["other"])
    reasons.append(f"profession {profession} (+{prof_pts})")

    source = (lead.get("source") or "other").lower()
    src_pts = config.SOURCE_POINTS.get(source, config.SOURCE_POINTS["other"])
    reasons.append(f"source {source} (+{src_pts})")

    age = lead.get("age")
    age_pts = 0
    if age:
        age = int(age)
        age_pts = 10 if 28 <= age <= 50 else 6 if 22 <= age <= 60 else 2
        reasons.append(f"age {age} (+{age_pts})")

    existing = int(lead.get("existing_investments") or 0)
    inv_pts = 10 if existing >= 1_000_000 else 6 if existing >= 200_000 else 2 if existing else 0
    if inv_pts:
        reasons.append(f"existing investments ₹{existing:,} (+{inv_pts})")

    consent_pts = 5 if lead.get("consent") else 0

    score = min(100, capacity + prof_pts + src_pts + age_pts + inv_pts + consent_pts)
    if score >= config.TIER_THRESHOLDS["A"]:
        tier = "A"
    elif score >= config.TIER_THRESHOLDS["B"]:
        tier = "B"
    else:
        tier = "C"
    return ScoreResult(score=score, tier=tier, sip_potential=potential, reasons=reasons)
