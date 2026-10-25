"""
WhatsApp nurture sequences, one per tier.

Rules baked in (SEBI/AMFI + WhatsApp policy):
  • Only message leads who gave consent (form tick-box / they messaged first).
  • No return guarantees; every message carries the standard risk line.
  • Every message offers STOP to opt out.
Placeholders: {name}, {advisor}, {brand}, {website}, {potential}, {goal}
"""

from . import config

RISK_LINE = "Mutual fund investments are subject to market risks, read all scheme related documents carefully."
FOOTER = "\n\n— {advisor}, {brand} ({arn})\n_" + RISK_LINE + "_\nReply STOP to unsubscribe."

# (step_key, day offset from lead creation, template)
SEQUENCES: dict[str, list[tuple[str, int, str]]] = {
    "A": [
        ("A0_welcome", 0,
         "Namaste {name} ji 🙏\nThanks for connecting with {brand}. Based on what you shared, a "
         "disciplined SIP of around ₹{potential:,}/month could move you meaningfully towards {goal}.\n"
         "Can I call you for 15 minutes today or tomorrow to build your free goal plan?"),
        ("A1_value", 1,
         "{name} ji, quick perspective: ₹50,000/month SIP for 15 years = ₹90 lakh invested. "
         "At an *assumed* 12% p.a. that could be about ₹2.5 Cr — but the real driver is staying "
         "invested. I've kept a slot for you — reply with a time that suits you."),
        ("A2_proof", 3,
         "{name} ji, most of my clients started exactly where you are — busy, unsure which fund. "
         "My job is a written plan: goals, asset allocation, tax efficiency, yearly review. "
         "Shall I visit you or would you like to come to our office in Jaora?"),
        ("A3_last", 7,
         "{name} ji, I'll stop following up after this one 🙂 If planning for {goal} is still on "
         "your list, just reply *PLAN* and I'll send a calendar slot."),
    ],
    "B": [
        ("B0_welcome", 0,
         "Namaste {name} ji 🙏 Thanks for reaching out to {brand}!\nWhat is your #1 money goal "
         "right now — child education, house, retirement, or tax saving? Reply with one word."),
        ("B1_calc", 2,
         "{name} ji, here's a free SIP calculator to see what your monthly savings can grow into: "
         "https://{website}\nWant me to do the math for your goal ({goal})? Reply *YES*."),
        ("B2_meeting", 5,
         "{name} ji, I'm doing free 20-minute goal-planning calls this week. "
         "Reply with a convenient day and I'll block it for you."),
        ("B3_last", 10,
         "Last nudge from my side, {name} ji 🙂 Whenever you're ready, reply *START* and "
         "we'll set up your SIP in 10 minutes, fully online."),
    ],
    "C": [
        ("C0_welcome", 0,
         "Namaste {name} ji 🙏 Welcome to {brand}! You'll get short, useful money tips from us "
         "— no spam. Free SIP calculator: https://{website}"),
        ("C1_tip", 4,
         "{name} ji, money tip: start small, increase every year. A ₹2,000 SIP stepped up by 10% "
         "each year builds far more than a flat SIP. Reply *STEPUP* to see your numbers."),
        ("C2_offer", 12,
         "{name} ji, want to start your first SIP? It takes 10 minutes online. "
         "Reply *START* and I'll guide you personally."),
    ],
}

# Messages for leads already further down the funnel
STAGE_NUDGES = {
    "meeting": ("Reminder 📅 {name} ji, looking forward to our meeting. Please keep your PAN, "
                "Aadhaar and a cancelled cheque handy so we can complete KYC in the same sitting."),
    "proposal": ("{name} ji, did you get a chance to go through the plan I shared? Happy to "
                 "clarify anything — once you're comfortable, the SIP can be registered online in 10 minutes."),
}

REFERRAL_ASK = (
    "{name} ji, thank you for trusting {brand} 🙏 Your SIP is now active.\n"
    "If you know 2 friends or family members who'd benefit from a free goal plan, just "
    "forward this number — I'll take care of them the same way."
)


HONORIFICS = {"dr", "dr.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "shri", "smt", "smt.", "ca", "adv", "adv."}


def first_name(full: str) -> str:
    parts = [p for p in (full or "").split() if p.lower() not in HONORIFICS]
    return parts[0] if parts else "Sir/Madam"


def render(template: str, lead: dict, with_footer: bool = True) -> str:
    fields = {
        "name": first_name(lead.get("name") or ""),
        "advisor": config.ADVISOR_NAME,
        "brand": config.BRAND,
        "website": config.WEBSITE,
        "arn": config.ARN_LINE,
        "potential": int(lead.get("sip_potential") or 0),
        "goal": lead.get("goal") or "your financial goals",
    }
    text = template + (FOOTER if with_footer else "")
    return text.format(**fields)
