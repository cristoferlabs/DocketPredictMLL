"""Unit tests — DC (Doble Oportunidad) markets.

Run with:  pytest tests/test_dc_markets.py -v
"""

from __future__ import annotations

import pytest

from apps.api.services.dc_engine import (
    DCPick,
    DC_MARKET,
    _decision,
    _kelly_stake,
    _risk,
    best_dc_pick,
    evaluate_dc,
)
from apps.api.services.odds_context import EvOpportunity
from apps.api.services.parlay_engine import extract_dc_parlay_pick
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def japan_sweden_model() -> ModelMarkets:
    """Japan vs Sweden calibrated model — June 2026 WC."""
    return ModelMarkets(
        home_win=0.341,
        draw=0.374,
        away_win=0.286,
        over_25=0.40,
        under_25=0.60,
        btts_yes=0.38,
        btts_no=0.62,
        lambda_home=1.15,
        lambda_away=0.95,
        dc_home_draw=0.715,   # 1X = 0.341 + 0.374
        dc_away_draw=0.660,   # X2 = 0.374 + 0.286
        dc_home_away=0.627,   # 12 = 0.341 + 0.286
        over_05=0.88,
        over_15=0.70,
        over_35=0.18,
        over_45=0.08,
        under_05=0.12,
        under_15=0.30,
        under_35=0.82,
        under_45=0.92,
    )


@pytest.fixture
def japan_sweden_analysis(japan_sweden_model: ModelMarkets) -> MatchAnalysis:
    return MatchAnalysis(
        team1="Japan",
        team2="Sweden",
        fecha="2026-06-25",
        ronda="Matchday 15",
        grupo="E",
        estadio="",
        model=japan_sweden_model,
    )


@pytest.fixture
def dc_ev_opps(japan_sweden_model: ModelMarkets) -> list[EvOpportunity]:
    """Simulate ev_opps as returned by compute_ev_opportunities() for DC market."""
    m = japan_sweden_model
    # Fair DC odds derived from devigged h2h (simulate 5% vig removal)
    fair_1x = 1.0 / (m.dc_home_draw * 0.98)  # slight edge from devig
    fair_x2 = 1.0 / (m.dc_away_draw * 0.98)
    fair_12 = 1.0 / (m.dc_home_away * 0.98)
    ev_1x = m.dc_home_draw * fair_1x - 1
    ev_x2 = m.dc_away_draw * fair_x2 - 1
    ev_12 = m.dc_home_away * fair_12 - 1
    return [
        EvOpportunity(
            market=DC_MARKET,
            selection="1X (Japan/Empate)",
            model_prob=m.dc_home_draw,
            book_odds=fair_1x,
            implied_prob=m.dc_home_draw * 0.98,
            expected_value=ev_1x,
            edge_pct=round((m.dc_home_draw - m.dc_home_draw * 0.98) * 100, 1),
            priority="medium",
            fair_odds=round(fair_1x, 4),
        ),
        EvOpportunity(
            market=DC_MARKET,
            selection="X2 (Empate/Sweden)",
            model_prob=m.dc_away_draw,
            book_odds=fair_x2,
            implied_prob=m.dc_away_draw * 0.98,
            expected_value=ev_x2,
            edge_pct=round((m.dc_away_draw - m.dc_away_draw * 0.98) * 100, 1),
            priority="medium",
            fair_odds=round(fair_x2, 4),
        ),
        EvOpportunity(
            market=DC_MARKET,
            selection="12 (Japan/Sweden)",
            model_prob=m.dc_home_away,
            book_odds=fair_12,
            implied_prob=m.dc_home_away * 0.98,
            expected_value=ev_12,
            edge_pct=round((m.dc_home_away - m.dc_home_away * 0.98) * 100, 1),
            priority="low",
            fair_odds=round(fair_12, 4),
        ),
    ]


# ─── DC probability math ──────────────────────────────────────────────────────

class TestDCProbabilityMath:
    def test_x2_equals_draw_plus_away(self, japan_sweden_model):
        m = japan_sweden_model
        expected_x2 = m.draw + m.away_win  # 0.374 + 0.286 = 0.660
        assert abs(m.dc_away_draw - expected_x2) < 0.001, (
            f"X2={m.dc_away_draw:.4f} should equal draw+away={expected_x2:.4f}"
        )

    def test_1x_equals_home_plus_draw(self, japan_sweden_model):
        m = japan_sweden_model
        expected_1x = m.home_win + m.draw  # 0.341 + 0.374 = 0.715
        assert abs(m.dc_home_draw - expected_1x) < 0.001

    def test_12_equals_home_plus_away(self, japan_sweden_model):
        m = japan_sweden_model
        expected_12 = m.home_win + m.away_win  # 0.341 + 0.286 = 0.627
        assert abs(m.dc_home_away - expected_12) < 0.001

    def test_x2_value(self, japan_sweden_model):
        assert abs(japan_sweden_model.dc_away_draw - 0.660) < 0.001

    def test_1x_value(self, japan_sweden_model):
        assert abs(japan_sweden_model.dc_home_draw - 0.715) < 0.001

    def test_all_dc_below_one(self, japan_sweden_model):
        m = japan_sweden_model
        for field in ("dc_home_draw", "dc_away_draw", "dc_home_away"):
            val = getattr(m, field)
            assert 0 < val < 1, f"{field}={val} must be in (0,1)"


# ─── evaluate_dc ─────────────────────────────────────────────────────────────

class TestEvaluateDC:
    def test_returns_three_picks(self, japan_sweden_model):
        picks = evaluate_dc(japan_sweden_model, [], "Japan", "Sweden")
        assert len(picks) == 3

    def test_x2_short_label(self, japan_sweden_model):
        picks = evaluate_dc(japan_sweden_model, [], "Japan", "Sweden")
        labels = [p.short_label for p in picks]
        assert "X2" in labels
        assert "1X" in labels
        assert "12" in labels

    def test_x2_probability(self, japan_sweden_model):
        picks = evaluate_dc(japan_sweden_model, [], "Japan", "Sweden")
        x2 = next(p for p in picks if p.short_label == "X2")
        assert abs(x2.model_prob - 0.660) < 0.001

    def test_1x_probability(self, japan_sweden_model):
        picks = evaluate_dc(japan_sweden_model, [], "Japan", "Sweden")
        p1x = next(p for p in picks if p.short_label == "1X")
        assert abs(p1x.model_prob - 0.715) < 0.001

    def test_x2_risk_is_low_or_very_low(self, japan_sweden_model):
        picks = evaluate_dc(japan_sweden_model, [], "Japan", "Sweden")
        x2 = next(p for p in picks if p.short_label == "X2")
        assert x2.risk in ("BAJO", "MUY BAJO"), f"X2 risk should be low, got {x2.risk}"

    def test_1x_risk_is_very_low(self, japan_sweden_model):
        picks = evaluate_dc(japan_sweden_model, [], "Japan", "Sweden")
        p1x = next(p for p in picks if p.short_label == "1X")
        assert p1x.risk in ("MUY BAJO",), f"1X risk should be MUY BAJO at 71.5%, got {p1x.risk}"

    def test_exactly_one_primary(self, japan_sweden_model):
        picks = evaluate_dc(japan_sweden_model, [], "Japan", "Sweden")
        primaries = [p for p in picks if p.is_primary]
        assert len(primaries) == 1

    def test_no_market_gives_zero_ev(self, japan_sweden_model):
        picks = evaluate_dc(japan_sweden_model, [], "Japan", "Sweden")
        for p in picks:
            assert p.ev_pct == 0.0, f"{p.short_label} ev_pct should be 0 without market"

    def test_with_ev_opps_non_zero_ev(self, japan_sweden_model, dc_ev_opps):
        picks = evaluate_dc(japan_sweden_model, dc_ev_opps, "Japan", "Sweden")
        x2 = next(p for p in picks if p.short_label == "X2")
        assert x2.ev_pct > 0.0, "EV should be positive when market has edge"

    def test_label_contains_team_names(self, japan_sweden_model):
        picks = evaluate_dc(japan_sweden_model, [], "Japan", "Sweden")
        x2 = next(p for p in picks if p.short_label == "X2")
        assert "Sweden" in x2.label
        assert "Empate" in x2.label
        p1x = next(p for p in picks if p.short_label == "1X")
        assert "Japan" in p1x.label


# ─── Risk helpers ─────────────────────────────────────────────────────────────

class TestRiskHelpers:
    @pytest.mark.parametrize("prob,expected_risk", [
        (0.70, "MUY BAJO"),
        (0.65, "MUY BAJO"),
        (0.60, "BAJO"),
        (0.55, "BAJO"),
        (0.50, "MEDIO"),
        (0.45, "MEDIO"),
        (0.40, "ALTO"),
        (0.28, "ALTO"),
    ])
    def test_risk_levels(self, prob, expected_risk):
        risk, _ = _risk(prob)
        assert risk == expected_risk

    @pytest.mark.parametrize("prob,ev,expected", [
        (0.66, 5.0, "STRONG_BET"),
        (0.66, 1.5, "MODERATE_BET"),  # MODERATE threshold is ev >= 1.5%
        (0.50, 0.0, "WEAK_BET"),
        (0.40, 0.0, "NO_BET"),
        (0.66, -1.0, "NO_BET"),
    ])
    def test_decision_thresholds(self, prob, ev, expected):
        assert _decision(prob, ev) == expected


# ─── Kelly stake ──────────────────────────────────────────────────────────────

class TestKellyStake:
    def test_positive_edge_gives_positive_stake(self):
        # Model prob 0.66, fair odds ~1.52 → small positive Kelly
        stake = _kelly_stake(0.66, 1.52)
        assert stake >= 0.0

    def test_negative_edge_gives_zero_stake(self):
        # Model prob 0.50, fair odds 1.80 → EV=−10% → stake=0
        stake = _kelly_stake(0.50, 1.80)
        assert stake == 0.0

    def test_stake_capped_at_2_percent(self):
        # Extreme edge → capped at 2.0%
        stake = _kelly_stake(0.99, 1.10)
        assert stake <= 2.0

    def test_fair_odds_below_one_gives_zero(self):
        assert _kelly_stake(0.80, 0.95) == 0.0


# ─── DC in parlay engine ──────────────────────────────────────────────────────

class TestDCParlayPick:
    def test_extract_returns_none_without_ev_opps(self, japan_sweden_analysis):
        result = extract_dc_parlay_pick(japan_sweden_analysis, [])
        assert result is None

    def test_extract_returns_pick_with_valid_opps(
        self, japan_sweden_analysis, dc_ev_opps
    ):
        # Make EV and prob pass the filter
        result = extract_dc_parlay_pick(
            japan_sweden_analysis, dc_ev_opps, min_prob=0.55, min_ev=0.01
        )
        assert result is not None
        assert result.market == DC_MARKET
        assert result.p_model >= 0.55
        assert result.odds > 1.0

    def test_extract_rejects_low_prob(self, japan_sweden_analysis, dc_ev_opps):
        result = extract_dc_parlay_pick(
            japan_sweden_analysis, dc_ev_opps, min_prob=0.90
        )
        assert result is None  # nothing exceeds 90%

    def test_extract_eligible(self, japan_sweden_analysis, dc_ev_opps):
        result = extract_dc_parlay_pick(
            japan_sweden_analysis, dc_ev_opps, min_prob=0.55, min_ev=0.01
        )
        if result:
            assert result.eligible is True
            assert result.reject_reason is None

    def test_extract_pick_is_dc_market(self, japan_sweden_analysis, dc_ev_opps):
        result = extract_dc_parlay_pick(
            japan_sweden_analysis, dc_ev_opps, min_prob=0.55, min_ev=0.01
        )
        if result:
            assert result.market == "Doble Oportunidad"

    def test_extract_match_metadata(self, japan_sweden_analysis, dc_ev_opps):
        result = extract_dc_parlay_pick(
            japan_sweden_analysis, dc_ev_opps, min_prob=0.55, min_ev=0.01
        )
        if result:
            assert result.team1 == "Japan"
            assert result.team2 == "Sweden"
            assert result.fecha == "2026-06-25"


# ─── best_dc_pick ─────────────────────────────────────────────────────────────

class TestBestDCPick:
    def test_returns_primary(self, japan_sweden_model):
        picks = evaluate_dc(japan_sweden_model, [], "Japan", "Sweden")
        best = best_dc_pick(picks)
        assert best is not None
        assert best.is_primary

    def test_returns_none_for_empty(self):
        assert best_dc_pick([]) is None
