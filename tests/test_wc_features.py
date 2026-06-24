"""Tests for World Cup feature engineering."""

from apps.worker.ml.wc_features import (
    compute_match_lambdas,
    estimate_team_xg,
    weighted_form_xg,
)


def test_weighted_form_recent_matches_weigh_more():
    form = [
        {"marcador": "3-0", "rival": "Brazil"},
        {"marcador": "0-1", "rival": "Weak Team"},
    ]
    elo = {"Brazil": 2000, "Weak Team": 1600}
    gf, w = weighted_form_xg(form, 1.2, elo, "Test FC")
    assert gf > 1.0
    assert w > 0


def test_estimate_team_xg_fallback_historical():
    profile = estimate_team_xg([], 1.4, {}, "Team A")
    assert profile.xg == 1.4
    assert profile.source == "historical"


def test_compute_match_lambdas_in_range():
    form_h = [{"marcador": "2-1", "rival": "B"}]
    form_a = [{"marcador": "1-0", "rival": "A"}]
    elo = {"Team A": 1800, "Team B": 1700, "A": 1500, "B": 1500}
    result = compute_match_lambdas(
        "Team A",
        "Team B",
        form_h,
        form_a,
        1.3,
        1.1,
        elo,
        [],
        [],
    )
    assert 0.5 <= result.lambda_home <= 4.0
    assert 0.5 <= result.lambda_away <= 4.0
