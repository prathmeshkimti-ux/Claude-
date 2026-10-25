# KIMTIMONEY Client Acquisition System

Goal: **10 new clients/week at ₹50,000/month SIP** (₹5L new SIP every week).

---

## 1. Reality check (read this first)

Run `python -m acquisition plan --budget 8000` to see the numbers:

| Funnel stage (per week) | Needed |
|---|---|
| Leads | ~510 |
| Qualified (income fits) | ~102 |
| Meetings | ~41 (about 7 every working day) |
| Proposals | ~29 |
| **Clients** | **10** |
| Paid-ad budget | ~₹1.17 L/month |

- A ₹50K SIP client earns **₹3–4L+/month**. In Jaora/Ratlam/Mandsaur that's doctors, CAs, big traders, business owners and NRIs. They are a small group, and **you don't reach them through Instagram ads**. You reach them through **referrals, seminars and personal networks**.
- At ₹8K/month on ads, paid leads alone will bring you **about 3 high-ticket clients a month**.
- 10 × ₹50K every week means adding ₹2.17 Cr of SIP a month. Your whole book today is ₹13L. Use it as a **north star, not a Week-1 target**.

**Realistic ramp-up:**

| Phase | Weekly target | Avg ticket | New SIP/month |
|---|---|---|---|
| Month 1–2 | 3 clients | ₹15K | ~₹2L |
| Month 3–6 | 5 clients | ₹25K | ~₹5.4L |
| Month 6–12 | 10 clients | ₹35–50K | ₹15–21L |

Even the Month 1–2 numbers would **grow your SIP book 6× faster** than the current +₹30K/month.

---

## 2. How the system works

```
 Lead sources ─▶ Google Form / Meta Lead Ads / website form ─▶ CSV
                                                              │
                                    python -m acquisition import leads.csv
                                                              │
                         ┌──────── Auto score (0–100) → Tier A / B / C ────────┐
                         │     income → SIP potential, profession, source,     │
                         │     age, existing investments, consent              │
                         └─────────────────────────────────────────────────────┘
                                                              │
                                  python -m acquisition daily   (every morning)
                                     │                                   │
                  WhatsApp queue (tier-specific drip,         Call list (Tier A/B,
                  click-to-send links, or Cloud API)          stale leads flagged)
                                     │                                   │
                       new → contacted → qualified → meeting → proposal → CLIENT
                                                                            │
                                                         Auto referral ask ─┘ (loop)
                                                              │
                                  python -m acquisition report   (every Sunday)
```

| Tier | Who | What the system does |
|---|---|---|
| **A** (score ≥70) | ₹25K+ SIP potential, referral/seminar | 4-message drip over 7 days. **You call them within 1 hour.** |
| **B** (45–69) | ₹10–25K potential | 4-message drip over 10 days, calculator link, then a call |
| **C** (<45) | Beginners | Education drip. Close online, no meeting needed |

---

## 3. Daily routine (45 min + calls)

| Time | Action | Command |
|---|---|---|
| 9:00 | Import yesterday's form and ad leads | `python -m acquisition import leads.csv` |
| 9:05 | Generate today's queue | `python -m acquisition daily` |
| 9:10 | Open `data/outbox/<date>_whatsapp.csv`, tap each link, send | — |
| 9:30–1:00 | Work the `<date>_calls.csv` list, top score first | — |
| After each call | Update the stage | `python -m acquisition stage 12 meeting` |
| SIP registered | Mark as won | `python -m acquisition stage 12 client --sip 50000` |
| Someone replies STOP | Opt them out | `python -m acquisition optout 98xxxxxxxx` |
| Sunday | Scorecard | `python -m acquisition report` |

---

## 4. Where ₹50K SIP clients actually come from (the 80/20)

| Channel | Cost/lead | Ticket size | Your weekly action |
|---|---|---|---|
| **Client referrals** | ₹0 | High | The system sends a referral ask to every new client. Also ask your top 30 existing clients personally |
| **Doctor/CA seminars** | ~₹150 | Very high | 1 per month: "Tax + retirement planning for doctors" at an IMA or CA association meet in Ratlam/Ujjain |
| **Business owner network** | ₹0–100 | Very high | Rotary/Lions/Jain Samaj/trader associations: short talk plus a free goal-plan offer |
| **CA partnerships** | ₹0 | High | Tie up with 3–5 CAs. Their clients with surplus cash = your SIP leads. Ask them to pass on leads, and **don't pay commission** (SEBI/AMFI rules) |
| **NRI families** | ₹0 | Very high | Families in Jaora with members in the Gulf/US. Run a video call with the NRI |
| **Meta lead ads** | ₹50–80 | Low–Mid | Keep the ₹8K here for B/C volume. Target 30–50 year olds within 50 km, interests: business, doctors |
| Instagram reels | ₹0 | Low–Mid | Brand and trust. They make the A-tier conversations easier |

**Budget split (₹8K/month):** ₹5K Meta lead ads · ₹2K one seminar (hall + tea) · ₹1K influencer/boost.

---

## 5. Lead form, copy-paste ready (Google Form / Meta Lead Ad)

Use these exact field names so `import` maps them automatically:

`name, phone, city, age, profession, monthly_income, existing_investments, goal, source, consent`

- **profession** options: doctor, ca, business_owner, trader, nri, govt_officer, it_professional, lawyer, salaried, retired, other
- **source**: set it as a hidden or prefilled value per campaign (`meta_ads`, `seminar`, `referral`, `influencer`, `website`, `corporate`)
- **consent** checkbox: *"I agree to receive investment information from KIMTIMONEY on WhatsApp/call."* **Required.** The system never messages a lead without it.

---

## 6. Compliance guardrails (already built in)

- Every message ends with the AMFI risk line and **"Reply STOP to unsubscribe"**
- No guaranteed returns. Projections say *"assumed 12%"*
- Only leads with consent get messages. Opt-outs are permanent
- Set your ARN in the footer: `export KM_ARN_LINE="ARN-XXXXXX | AMFI-registered MFD"`
- Don't pay referral fees to non-registered people. Use thank-you gestures (a gift hamper, a free review) instead

---

## 7. Optional: full auto-send with the WhatsApp Cloud API

```bash
export WA_CLOUD_TOKEN=...            # Meta Business → WhatsApp → API setup
export WA_PHONE_NUMBER_ID=...
python -m acquisition daily --send
```
Meta only allows free-text messages within 24h of the lead's last message. Cold drips need **approved templates**, so get the welcome/drip templates approved first. Until then, the click-to-send links work with your normal WhatsApp Business app.

To run it every morning automatically (Linux/Mac cron):
```
0 9 * * * cd /path/to/repo && python3 -m acquisition daily >> acquisition/data/daily.log 2>&1
```

---

## 8. Tune it after 4 weeks

Open `acquisition/config.py` and replace `FUNNEL_RATES` and `COST_PER_LEAD` with your real numbers from `report`. Then run `plan` again. It will tell you exactly how many leads, meetings and rupees you need for your target.
