"""
Reverse-funnel planner: start from the client target and work back to leads and budget.
"""

from dataclasses import dataclass

from . import config


@dataclass
class FunnelPlan:
    clients: float
    proposals: float
    meetings: float
    qualified: float
    leads: float
    weekly_budget: float
    monthly_budget: float
    new_sip_per_month: float
    yearly_trail_after_12m: float

    def as_table(self) -> str:
        rows = [
            ("Leads needed / week", f"{self.leads:,.0f}"),
            ("Qualified leads / week", f"{self.qualified:,.0f}"),
            ("Meetings / week", f"{self.meetings:,.0f}  (~{self.meetings / 6:.1f} per working day)"),
            ("Proposals / week", f"{self.proposals:,.0f}"),
            ("New clients / week", f"{self.clients:,.0f}"),
            ("Ad budget / week", f"₹{self.weekly_budget:,.0f}"),
            ("Ad budget / month", f"₹{self.monthly_budget:,.0f}"),
            ("New SIP added / month", f"₹{self.new_sip_per_month:,.0f}"),
            ("Approx. yearly trail after 12 months", f"₹{self.yearly_trail_after_12m:,.0f}"),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"  {k.ljust(width)}  {v}" for k, v in rows)


def plan(
    clients_per_week: float = config.TARGET_CLIENTS_PER_WEEK,
    sip_per_client: float = config.TARGET_SIP_PER_CLIENT,
    channel_mix: dict[str, float] | None = None,
    rates: dict[str, float] | None = None,
    trail_rate: float = 0.0092,
) -> FunnelPlan:
    r = rates or config.FUNNEL_RATES
    mix = channel_mix or {"meta_ads": 0.5, "influencer": 0.2, "referral": 0.2, "seminar": 0.1}

    proposals = clients_per_week / r["proposal_to_client"]
    meetings = proposals / r["meeting_to_proposal"]
    qualified = meetings / r["qualified_to_meeting"]
    leads = qualified / r["lead_to_qualified"]

    blended_cpl = sum(config.COST_PER_LEAD.get(ch, 50) * share for ch, share in mix.items())
    weekly_budget = leads * blended_cpl
    new_sip_month = clients_per_week * 52 / 12 * sip_per_client

    # SIPs added evenly over 12 months → average ~6.5 instalments each by month 12
    aum_after_12m = new_sip_month * 12 * 6.5
    return FunnelPlan(
        clients=clients_per_week,
        proposals=proposals,
        meetings=meetings,
        qualified=qualified,
        leads=leads,
        weekly_budget=weekly_budget,
        monthly_budget=weekly_budget * 52 / 12,
        new_sip_per_month=new_sip_month,
        yearly_trail_after_12m=aum_after_12m * trail_rate,
    )


def max_clients_for_budget(monthly_budget: float, **kwargs) -> float:
    """How many clients/week a given monthly ad budget supports at current rates."""
    one = plan(clients_per_week=1, **kwargs)
    return monthly_budget / one.monthly_budget if one.monthly_budget else 0.0
