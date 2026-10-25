"""
Daily engine: ingest leads, score them, queue today's WhatsApp messages,
build the call list and the weekly report.
"""

import csv
import json
import re
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

from . import config, db
from .scoring import score_lead
from .sequences import REFERRAL_ASK, SEQUENCES, STAGE_NUDGES, render

LEAD_FIELDS = [
    "name", "phone", "city", "age", "profession", "monthly_income",
    "existing_investments", "goal", "source", "referred_by", "consent", "notes",
]


def normalize_phone(raw: str) -> str:
    digits = re.sub(r"\D", "", str(raw))
    if len(digits) == 10:
        digits = "91" + digits
    if len(digits) == 11 and digits.startswith("0"):
        digits = "91" + digits[1:]
    if len(digits) < 11:
        raise ValueError(f"Invalid phone number: {raw!r}")
    return digits


def _to_int(value) -> int:
    if value in (None, ""):
        return 0
    return int(float(str(value).replace(",", "").replace("₹", "").strip()))


def _to_bool(value) -> int:
    return 1 if str(value).strip().lower() in {"1", "yes", "y", "true", "haan", "ha"} else 0


def add_lead(conn, data: dict) -> int | None:
    """Insert or update a lead (dedup by phone). Returns lead id."""
    lead = {k: data.get(k, "") for k in LEAD_FIELDS}
    lead["phone"] = normalize_phone(lead["phone"])
    for k in ("age", "monthly_income", "existing_investments"):
        lead[k] = _to_int(lead[k])
    lead["consent"] = _to_bool(lead["consent"])
    lead["profession"] = (lead["profession"] or "other").strip().lower().replace(" ", "_")
    lead["source"] = (lead["source"] or "other").strip().lower().replace(" ", "_")

    result = score_lead(lead)
    existing = conn.execute("SELECT id FROM leads WHERE phone = ?", (lead["phone"],)).fetchone()
    if existing:
        conn.execute(
            """UPDATE leads SET name=?, city=?, age=?, profession=?, monthly_income=?,
               existing_investments=?, goal=?, source=?, referred_by=?, consent=MAX(consent, ?),
               score=?, tier=?, sip_potential=? WHERE id=?""",
            (lead["name"], lead["city"], lead["age"], lead["profession"], lead["monthly_income"],
             lead["existing_investments"], lead["goal"], lead["source"], lead["referred_by"],
             lead["consent"], result.score, result.tier, result.sip_potential, existing["id"]),
        )
        return existing["id"]

    ts = db.now()
    cur = conn.execute(
        """INSERT INTO leads (name, phone, city, age, profession, monthly_income,
           existing_investments, goal, source, referred_by, consent, notes,
           score, tier, sip_potential, stage, created_at, stage_changed_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'new',?,?)""",
        (lead["name"], lead["phone"], lead["city"], lead["age"], lead["profession"],
         lead["monthly_income"], lead["existing_investments"], lead["goal"], lead["source"],
         lead["referred_by"], lead["consent"], lead["notes"],
         result.score, result.tier, result.sip_potential, ts, ts),
    )
    return cur.lastrowid


def import_csv(conn, path: str | Path) -> tuple[int, list[str]]:
    """Import leads from a CSV export (Google Form / Meta Lead Ads / website)."""
    count, errors = 0, []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            row = {k.strip().lower().replace(" ", "_"): (v or "").strip() for k, v in row.items() if k}
            try:
                add_lead(conn, row)
                count += 1
            except (ValueError, KeyError) as exc:
                errors.append(f"row {i}: {exc}")
    return count, errors


# ── Messaging ───────────────────────────────────────────────────────────────

def wa_link(phone: str, text: str) -> str:
    return f"https://wa.me/{phone}?text={urllib.parse.quote(text)}"


def send_whatsapp_cloud(phone: str, text: str) -> bool:
    """Send via WhatsApp Cloud API if configured.

    Note: outside the 24h customer-service window Meta only allows pre-approved
    template messages; free text here works for leads who messaged you first.
    """
    if not (config.WA_TOKEN and config.WA_PHONE_ID):
        return False
    req = urllib.request.Request(
        f"https://graph.facebook.com/v20.0/{config.WA_PHONE_ID}/messages",
        data=json.dumps({
            "messaging_product": "whatsapp", "to": phone,
            "type": "text", "text": {"body": text},
        }).encode(),
        headers={"Authorization": f"Bearer {config.WA_TOKEN}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except OSError:
        return False


def due_messages(conn, today: date | None = None) -> list[dict]:
    """Next due sequence step per consenting, active lead (one message per lead per day)."""
    today = today or date.today()
    sent = {(r["lead_id"], r["step_key"]) for r in conn.execute("SELECT lead_id, step_key FROM messages")}
    out = []
    leads = conn.execute(
        "SELECT * FROM leads WHERE consent = 1 AND opted_out = 0 AND stage NOT IN ('client','lost')"
    ).fetchall()
    for lead in leads:
        lead = dict(lead)
        age_days = (today - datetime.fromisoformat(lead["created_at"]).date()).days
        if lead["stage"] in STAGE_NUDGES:
            key = f"nudge_{lead['stage']}_{lead['stage_changed_at'][:10]}"
            if (lead["id"], key) not in sent:
                out.append({"lead": lead, "step_key": key,
                            "text": render(STAGE_NUDGES[lead["stage"]], lead)})
            continue
        if lead["stage"] not in ("new", "contacted"):
            continue
        for key, offset, template in SEQUENCES[lead["tier"]]:
            if (lead["id"], key) in sent:
                continue
            if age_days >= offset:
                out.append({"lead": lead, "step_key": key, "text": render(template, lead)})
            break
    # also queue one referral ask for every new client
    for lead in conn.execute("SELECT * FROM leads WHERE stage = 'client' AND opted_out = 0"):
        lead = dict(lead)
        if (lead["id"], "referral_ask") not in sent:
            out.append({"lead": lead, "step_key": "referral_ask", "text": render(REFERRAL_ASK, lead)})
    out.sort(key=lambda m: -m["lead"]["score"])
    return out


def call_list(conn, today: date | None = None) -> list[dict]:
    """Tier-A/B leads that need a human call today, highest score first."""
    today = today or date.today()
    rows = []
    for lead in conn.execute(
        "SELECT * FROM leads WHERE opted_out = 0 AND stage NOT IN ('client','lost') "
        "AND tier IN ('A','B') ORDER BY score DESC"
    ):
        lead = dict(lead)
        idle = (today - datetime.fromisoformat(lead["stage_changed_at"]).date()).days
        limit = config.STALE_AFTER_DAYS.get(lead["stage"], 3)
        if lead["stage"] == "new" or idle >= limit:
            action = {
                "new": "First call within 1 hour of lead — speed wins",
                "contacted": "Qualify: income, goals, existing investments",
                "qualified": "Book meeting (office / home visit / video)",
                "meeting": "Confirm meeting; carry KYC checklist",
                "proposal": "Close: address objections, register SIP",
            }[lead["stage"]]
            rows.append({**lead, "idle_days": idle, "action": action})
    return rows


def run_daily(conn, today: date | None = None, send: bool = False) -> dict:
    today = today or date.today()
    calls = call_list(conn, today)  # before messaging moves leads out of 'new'
    messages = due_messages(conn, today)
    config.OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    outbox = config.OUTBOX_DIR / f"{today.isoformat()}_whatsapp.csv"
    sent_api = 0
    with open(outbox, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lead_id", "name", "tier", "score", "step", "click_to_send", "message"])
        for m in messages:
            lead = m["lead"]
            channel = "link"
            if send and send_whatsapp_cloud(lead["phone"], m["text"]):
                channel, sent_api = "cloud_api", sent_api + 1
            w.writerow([lead["id"], lead["name"], lead["tier"], lead["score"], m["step_key"],
                        wa_link(lead["phone"], m["text"]), m["text"]])
            conn.execute(
                "INSERT OR IGNORE INTO messages (lead_id, step_key, sent_at, channel) VALUES (?,?,?,?)",
                (lead["id"], m["step_key"], db.now(), channel),
            )
            if lead["stage"] == "new":
                db.set_stage(conn, lead["id"], "contacted")

    calls_file = config.OUTBOX_DIR / f"{today.isoformat()}_calls.csv"
    with open(calls_file, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lead_id", "name", "phone", "tier", "score", "sip_potential", "stage", "idle_days", "action"])
        for c in calls:
            w.writerow([c["id"], c["name"], c["phone"], c["tier"], c["score"], c["sip_potential"],
                        c["stage"], c["idle_days"], c["action"]])
    return {"messages": len(messages), "sent_via_api": sent_api, "calls": calls,
            "outbox": str(outbox), "calls_file": str(calls_file)}


def opt_out(conn, phone: str) -> None:
    conn.execute("UPDATE leads SET opted_out = 1 WHERE phone = ?", (normalize_phone(phone),))


# ── Reporting ───────────────────────────────────────────────────────────────

def weekly_report(conn, week_ending: date | None = None) -> dict:
    end = week_ending or date.today()
    start = end - timedelta(days=6)
    s, e = start.isoformat(), (end + timedelta(days=1)).isoformat()

    new_leads = conn.execute(
        "SELECT source, tier, COUNT(*) n FROM leads WHERE created_at >= ? AND created_at < ? "
        "GROUP BY source, tier", (s, e)).fetchall()
    moves = conn.execute(
        "SELECT to_stage, COUNT(DISTINCT lead_id) n FROM stage_history "
        "WHERE changed_at >= ? AND changed_at < ? GROUP BY to_stage", (s, e)).fetchall()
    won = conn.execute(
        "SELECT l.name, l.sip_amount, l.source FROM stage_history h JOIN leads l ON l.id = h.lead_id "
        "WHERE h.to_stage = 'client' AND h.changed_at >= ? AND h.changed_at < ?", (s, e)).fetchall()
    pipeline = conn.execute(
        "SELECT stage, COUNT(*) n, SUM(sip_potential) pot FROM leads "
        "WHERE stage NOT IN ('client','lost') GROUP BY stage").fetchall()

    by_source: dict[str, int] = {}
    by_tier: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    for r in new_leads:
        by_source[r["source"]] = by_source.get(r["source"], 0) + r["n"]
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + r["n"]
    stage_moves = {r["to_stage"]: r["n"] for r in moves}
    new_sip = sum(r["sip_amount"] or 0 for r in won)
    return {
        "week": f"{start.isoformat()} → {end.isoformat()}",
        "new_leads": sum(by_source.values()),
        "leads_by_source": by_source,
        "leads_by_tier": by_tier,
        "stage_moves": stage_moves,
        "clients_won": len(won),
        "new_sip_amount": new_sip,
        "avg_ticket": new_sip // len(won) if won else 0,
        "target_clients": config.TARGET_CLIENTS_PER_WEEK,
        "target_sip": config.TARGET_CLIENTS_PER_WEEK * config.TARGET_SIP_PER_CLIENT,
        "won": [dict(r) for r in won],
        "pipeline": {r["stage"]: {"count": r["n"], "potential": r["pot"] or 0} for r in pipeline},
    }
