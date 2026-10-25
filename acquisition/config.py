"""
KIMTIMONEY client acquisition — tunable settings.

Edit the numbers here as real data comes in; every other module reads from this file.
"""

import os
from pathlib import Path

BRAND = "KIMTIMONEY"
ADVISOR_NAME = os.getenv("KM_ADVISOR_NAME", "Prathmesh")
ADVISOR_PHONE = os.getenv("KM_ADVISOR_PHONE", "")  # used in message footers
WEBSITE = "kimtimoney.com"
ARN_LINE = os.getenv("KM_ARN_LINE", "AMFI-registered Mutual Fund Distributor")

DB_PATH = Path(os.getenv("KM_DB_PATH", Path(__file__).parent / "data" / "leads.db"))
OUTBOX_DIR = Path(os.getenv("KM_OUTBOX_DIR", Path(__file__).parent / "data" / "outbox"))

# ── Weekly goal ─────────────────────────────────────────────────────────────
TARGET_CLIENTS_PER_WEEK = 10
TARGET_SIP_PER_CLIENT = 50_000          # ₹/month

# ── Funnel conversion assumptions (replace with your own numbers after 4 weeks)
# Stage-to-stage rates for a HIGH-ticket (₹25K+ SIP) lead.
FUNNEL_RATES = {
    "lead_to_qualified": 0.20,   # filled form / replied + income fits
    "qualified_to_meeting": 0.40,  # agreed to a call / office visit
    "meeting_to_proposal": 0.70,  # goal-based plan shared
    "proposal_to_client": 0.35,   # first SIP registered
}
COST_PER_LEAD = {                # ₹, rough paid-channel averages for Tier-3 MP
    "meta_ads": 60,
    "google_ads": 120,
    "influencer": 40,
    "referral": 0,
    "seminar": 150,
    "organic": 0,
    "website": 0,
    "corporate": 100,
    "other": 50,
}

# ── Lead scoring ────────────────────────────────────────────────────────────
# Investable surplus = what a lead can realistically put into SIP each month.
SURPLUS_RATIO_BY_INCOME = [      # (monthly income ≥ ₹, share of income investable)
    (300_000, 0.30),
    (150_000, 0.25),
    (75_000, 0.20),
    (40_000, 0.15),
    (0, 0.10),
]
PROFESSION_POINTS = {
    "doctor": 15, "ca": 15, "business_owner": 15, "trader": 12,
    "nri": 15, "govt_officer": 10, "it_professional": 12, "lawyer": 12,
    "salaried": 6, "farmer_large": 8, "retired": 6, "other": 3,
}
SOURCE_POINTS = {
    "referral": 20, "seminar": 15, "corporate": 12, "website": 10,
    "influencer": 8, "google_ads": 8, "meta_ads": 6, "organic": 6, "other": 3,
}
TIER_THRESHOLDS = {"A": 70, "B": 45}  # score ≥ 70 → A, ≥ 45 → B, else C

# ── Pipeline stages (order matters) ─────────────────────────────────────────
STAGES = ["new", "contacted", "qualified", "meeting", "proposal", "client", "lost"]
# Days without a stage move before a lead is flagged as stale
STALE_AFTER_DAYS = {"new": 1, "contacted": 3, "qualified": 3, "meeting": 2, "proposal": 4}

# WhatsApp Cloud API (optional). Without a token the system writes click-to-chat links.
WA_TOKEN = os.getenv("WA_CLOUD_TOKEN", "")
WA_PHONE_ID = os.getenv("WA_PHONE_NUMBER_ID", "")
