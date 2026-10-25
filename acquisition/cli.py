"""
KIMTIMONEY client acquisition CLI.

  python -m acquisition plan                       # reverse funnel for 10 × ₹50K/week
  python -m acquisition plan --budget 8000         # what your current budget can deliver
  python -m acquisition import leads.csv           # import Google Form / Meta lead CSV
  python -m acquisition add --name "Rahul Jain" --phone 98xxxxxxxx --income 200000 \
        --profession doctor --source referral --consent yes --goal "child education"
  python -m acquisition daily [--send]             # today's WhatsApp queue + call list
  python -m acquisition stage 12 meeting           # move lead 12 to a stage
  python -m acquisition stage 12 client --sip 50000
  python -m acquisition optout 98xxxxxxxx
  python -m acquisition leads [--tier A]
  python -m acquisition report                     # weekly scorecard
"""

import argparse
import sys

from . import config, db, engine, funnel


def _inr(n) -> str:
    return f"₹{int(n or 0):,}"


def cmd_plan(args) -> None:
    p = funnel.plan(clients_per_week=args.clients, sip_per_client=args.sip)
    print(f"\nTarget: {args.clients:g} clients/week × {_inr(args.sip)} SIP\n")
    print(p.as_table())
    if args.budget:
        cap = funnel.max_clients_for_budget(args.budget)
        print(f"\nWith a {_inr(args.budget)}/month ad budget at current rates you can expect "
              f"≈ {cap:.1f} high-ticket clients/week ({cap * 52 / 12:.1f}/month) from paid leads.")
        print("Referrals, seminars and corporate tie-ups (₹0–150/lead) are how you close the gap.")


def cmd_import(args) -> None:
    with db.connect() as conn:
        n, errors = engine.import_csv(conn, args.path)
    print(f"Imported/updated {n} leads.")
    for e in errors:
        print("  skipped", e)


def cmd_add(args) -> None:
    with db.connect() as conn:
        lead_id = engine.add_lead(conn, {
            "name": args.name, "phone": args.phone, "city": args.city, "age": args.age,
            "profession": args.profession, "monthly_income": args.income,
            "existing_investments": args.existing, "goal": args.goal, "source": args.source,
            "referred_by": args.referred_by, "consent": args.consent, "notes": args.notes,
        })
        row = conn.execute("SELECT score, tier, sip_potential FROM leads WHERE id=?", (lead_id,)).fetchone()
    print(f"Lead #{lead_id}: score {row['score']} · tier {row['tier']} · "
          f"SIP potential {_inr(row['sip_potential'])}/mo")


def cmd_daily(args) -> None:
    with db.connect() as conn:
        res = engine.run_daily(conn, send=args.send)
    calls = res["calls"]
    print(f"\nWhatsApp messages queued: {res['messages']} (sent via API: {res['sent_via_api']})")
    print(f"  → open {res['outbox']} and tap each click_to_send link")
    print(f"Calls to make today: {len(calls)}  → {res['calls_file']}\n")
    for c in calls[:15]:
        print(f"  [{c['tier']}{c['score']:>3}] #{c['id']:<4} {c['name'][:20]:<20} +{c['phone']}  "
              f"{c['stage']:<9} {_inr(c['sip_potential']):>9}/mo  {c['action']}")


def cmd_stage(args) -> None:
    with db.connect() as conn:
        db.set_stage(conn, args.lead_id, args.stage, sip_amount=args.sip)
    print(f"Lead #{args.lead_id} → {args.stage}")


def cmd_optout(args) -> None:
    with db.connect() as conn:
        engine.opt_out(conn, args.phone)
    print("Opted out. No further messages will be queued.")


def cmd_leads(args) -> None:
    q, params = "SELECT * FROM leads WHERE 1=1", []
    if args.tier:
        q, params = q + " AND tier = ?", params + [args.tier.upper()]
    if args.stage:
        q, params = q + " AND stage = ?", params + [args.stage]
    with db.connect() as conn:
        rows = conn.execute(q + " ORDER BY score DESC LIMIT ?", params + [args.limit]).fetchall()
    for r in rows:
        print(f"  #{r['id']:<4} [{r['tier']}{r['score']:>3}] {r['name'][:22]:<22} {r['stage']:<9} "
              f"{r['source']:<11} {_inr(r['sip_potential']):>9}/mo")


def cmd_report(args) -> None:
    with db.connect() as conn:
        r = engine.weekly_report(conn)
    pct = 100 * r["clients_won"] / r["target_clients"] if r["target_clients"] else 0
    print(f"\n📊 {config.BRAND} weekly scorecard  ({r['week']})")
    print(f"  Clients won      {r['clients_won']} / {r['target_clients']}  ({pct:.0f}%)")
    print(f"  New SIP          {_inr(r['new_sip_amount'])} / {_inr(r['target_sip'])}")
    print(f"  Avg ticket       {_inr(r['avg_ticket'])}")
    print(f"  New leads        {r['new_leads']}  tiers {r['leads_by_tier']}")
    print(f"  By source        {r['leads_by_source']}")
    print(f"  Stage moves      {r['stage_moves']}")
    print("  Open pipeline:")
    for stage in config.STAGES:
        if stage in r["pipeline"]:
            p = r["pipeline"][stage]
            print(f"    {stage:<10} {p['count']:>4} leads   potential {_inr(p['potential'])}/mo")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="acquisition", description=f"{config.BRAND} client acquisition")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="reverse-funnel calculator")
    p.add_argument("--clients", type=float, default=config.TARGET_CLIENTS_PER_WEEK)
    p.add_argument("--sip", type=float, default=config.TARGET_SIP_PER_CLIENT)
    p.add_argument("--budget", type=float, help="your monthly ad budget in ₹")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("import", help="import leads from CSV")
    p.add_argument("path")
    p.set_defaults(fn=cmd_import)

    p = sub.add_parser("add", help="add one lead")
    p.add_argument("--name", required=True)
    p.add_argument("--phone", required=True)
    p.add_argument("--city", default="")
    p.add_argument("--age", type=int)
    p.add_argument("--profession", default="other")
    p.add_argument("--income", type=int, default=0, help="monthly income ₹")
    p.add_argument("--existing", type=int, default=0, help="existing investments ₹")
    p.add_argument("--goal", default="")
    p.add_argument("--source", default="other")
    p.add_argument("--referred-by", default="")
    p.add_argument("--consent", default="no")
    p.add_argument("--notes", default="")
    p.set_defaults(fn=cmd_add)

    p = sub.add_parser("daily", help="build today's WhatsApp queue and call list")
    p.add_argument("--send", action="store_true", help="send via WhatsApp Cloud API if configured")
    p.set_defaults(fn=cmd_daily)

    p = sub.add_parser("stage", help="move a lead to a pipeline stage")
    p.add_argument("lead_id", type=int)
    p.add_argument("stage", choices=config.STAGES)
    p.add_argument("--sip", type=int, help="SIP amount when marking as client")
    p.set_defaults(fn=cmd_stage)

    p = sub.add_parser("optout", help="stop messaging a phone number")
    p.add_argument("phone")
    p.set_defaults(fn=cmd_optout)

    p = sub.add_parser("leads", help="list leads")
    p.add_argument("--tier")
    p.add_argument("--stage")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(fn=cmd_leads)

    p = sub.add_parser("report", help="weekly scorecard")
    p.set_defaults(fn=cmd_report)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
