"""Tests for the KIMTIMONEY client acquisition system."""

from datetime import date, timedelta

import pytest

from acquisition import config, db, engine, funnel
from acquisition.scoring import estimate_sip_potential, score_lead
from acquisition.sequences import RISK_LINE, first_name


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTBOX_DIR", tmp_path / "outbox")
    with db.connect(tmp_path / "leads.db") as c:
        yield c


def _lead(**kw):
    base = {"name": "Test Lead", "phone": "9826000001", "monthly_income": 200000,
            "profession": "doctor", "source": "referral", "age": 35, "consent": "yes"}
    base.update(kw)
    return base


class TestScoring:
    def test_high_income_referral_is_tier_a(self):
        r = score_lead(_lead(monthly_income=350000, existing_investments=2_000_000, consent=1))
        assert r.tier == "A"
        assert r.sip_potential == 105000

    def test_low_income_cold_lead_is_tier_c(self):
        r = score_lead(_lead(monthly_income=25000, profession="other", source="meta_ads", age=21))
        assert r.tier == "C"

    def test_potential_scales_with_income(self):
        assert estimate_sip_potential(50000) < estimate_sip_potential(200000)
        assert estimate_sip_potential(0) == 0


class TestEngine:
    def test_phone_normalisation(self):
        assert engine.normalize_phone("98260 00001") == "919826000001"
        assert engine.normalize_phone("09826000001") == "919826000001"
        with pytest.raises(ValueError):
            engine.normalize_phone("12345")

    def test_dedup_by_phone(self, conn):
        a = engine.add_lead(conn, _lead())
        b = engine.add_lead(conn, _lead(name="Updated", phone="+91 98260-00001"))
        assert a == b
        assert conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0] == 1

    def test_no_consent_means_no_message(self, conn):
        engine.add_lead(conn, _lead(consent="no"))
        assert engine.due_messages(conn) == []

    def test_opt_out_stops_messages(self, conn):
        engine.add_lead(conn, _lead())
        engine.opt_out(conn, "9826000001")
        assert engine.due_messages(conn) == []

    def test_messages_carry_compliance_footer(self, conn):
        engine.add_lead(conn, _lead())
        msgs = engine.due_messages(conn)
        assert len(msgs) == 1
        assert RISK_LINE in msgs[0]["text"] and "STOP" in msgs[0]["text"]

    def test_sequence_advances_one_step_per_day(self, conn):
        engine.add_lead(conn, _lead())
        today = date.today()
        res = engine.run_daily(conn, today=today)
        assert res["messages"] == 1
        # same day again → nothing new
        assert engine.run_daily(conn, today=today)["messages"] == 0
        # next day → step 2 of tier A sequence
        nxt = engine.due_messages(conn, today + timedelta(days=1))
        assert [m["step_key"] for m in nxt] == ["A1_value"]

    def test_new_lead_on_call_list_and_moves_to_contacted(self, conn):
        lead_id = engine.add_lead(conn, _lead())
        res = engine.run_daily(conn)
        assert [c["id"] for c in res["calls"]] == [lead_id]
        stage = conn.execute("SELECT stage FROM leads WHERE id=?", (lead_id,)).fetchone()[0]
        assert stage == "contacted"

    def test_client_gets_referral_ask_and_counts_in_report(self, conn):
        lead_id = engine.add_lead(conn, _lead())
        db.set_stage(conn, lead_id, "client", sip_amount=50000)
        assert [m["step_key"] for m in engine.due_messages(conn)] == ["referral_ask"]
        r = engine.weekly_report(conn)
        assert r["clients_won"] == 1 and r["new_sip_amount"] == 50000

    def test_import_csv(self, conn, tmp_path):
        p = tmp_path / "leads.csv"
        p.write_text("Name,Phone,Monthly Income,Source,Consent\n"
                     "A One,9826000011,100000,meta_ads,yes\nBad,123,0,x,no\n")
        n, errors = engine.import_csv(conn, p)
        assert n == 1 and len(errors) == 1

    def test_unknown_stage_rejected(self, conn):
        lead_id = engine.add_lead(conn, _lead())
        with pytest.raises(ValueError):
            db.set_stage(conn, lead_id, "won")


class TestFunnel:
    def test_reverse_funnel_math(self):
        p = funnel.plan(clients_per_week=10, sip_per_client=50000)
        r = config.FUNNEL_RATES
        expected_leads = 10 / r["proposal_to_client"] / r["meeting_to_proposal"] \
            / r["qualified_to_meeting"] / r["lead_to_qualified"]
        assert p.leads == pytest.approx(expected_leads)
        assert p.new_sip_per_month == pytest.approx(10 * 52 / 12 * 50000)

    def test_budget_capacity_is_linear(self):
        assert funnel.max_clients_for_budget(16000) == pytest.approx(
            2 * funnel.max_clients_for_budget(8000))


def test_first_name_skips_honorifics():
    assert first_name("Dr. Amit Jain") == "Amit"
    assert first_name("") == "Sir/Madam"
