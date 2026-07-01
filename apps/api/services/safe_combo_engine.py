"""Safe intra-match combination engine — exact Poisson joint probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from apps.api.services.worldcup_engine import ModelMarkets

if TYPE_CHECKING:
    from apps.worker.ml.poisson_live import LivePoissonResult

_VIG_FACTOR = 1.05  # 5% vig buffer → min market odds for +EV
_MIN_COMBO_PROB = 0.30
_MIN_LEG_PROB = 0.01
_MAX_RESULTS = 8  # increased from 5 to surface more strong bets


@dataclass
class ScoreBreakdown:
    prob_base: int        # combo_prob × 100 (main component)
    legs_bonus: int       # bonus for individually strong legs
    corr_bonus: int       # Poisson joint vs independence (positive = co-occurrence boost)
    risk_penalty: int     # penalty for low combo probability (< 45%)
    total: int            # final score [0, 100] — matches SafeCombo.score
    live_bonus: int = 0   # live signal alignment bonus [−5, +10]

    def detail_lines(self) -> list[str]:
        rows: list[tuple[str, int]] = [
            ("Prob conjunta", self.prob_base),
            ("Piernas fuertes", self.legs_bonus),
            ("Correlación", self.corr_bonus),
            ("Riesgo", self.risk_penalty),
        ]
        if self.live_bonus:
            rows.append(("Señal live", self.live_bonus))
        lines = []
        for label, val in rows:
            if val != 0:
                lines.append(f"  {label:<16} {val:+4d}")
        lines.append(f"  {'─'*22}")
        lines.append(f"  {'Total':<16} {self.total:4d}/100")
        return lines

    def short_str(self) -> str:
        parts = [f"base {self.prob_base}"]
        if self.legs_bonus:
            parts.append(f"piernas +{self.legs_bonus}")
        if self.corr_bonus:
            sign = "+" if self.corr_bonus >= 0 else ""
            parts.append(f"corr {sign}{self.corr_bonus}")
        if self.risk_penalty:
            parts.append(f"riesgo {self.risk_penalty}")
        if self.live_bonus:
            sign = "+" if self.live_bonus >= 0 else ""
            parts.append(f"live {sign}{self.live_bonus}")
        return " · ".join(parts)


@dataclass
class SafeCombo:
    leg1_label: str
    leg2_label: str
    leg1_prob: float
    leg2_prob: float
    combo_prob: float
    fair_odds: float
    market_min_odds: float
    score: int
    score_breakdown: ScoreBreakdown
    decision: str       # "STRONG_BET", "MODERATE_BET", "WEAK_BET"
    risk: str           # "BAJO", "MEDIO", "ALTO"
    recommended: bool
    label1_display: str
    label2_display: str


def _build_matrix(lambda_home: float, lambda_away: float) -> np.ndarray:
    from apps.worker.ml.poisson import predict_match, PoissonConfig
    pred = predict_match(lambda_home, lambda_away, PoissonConfig(max_goals=10))
    return pred.score_matrix


def _joint(matrix: np.ndarray, cond) -> float:
    mg = matrix.shape[0] - 1
    return float(sum(
        matrix[i, j]
        for i in range(mg + 1)
        for j in range(mg + 1)
        if cond(i, j)
    ))


def _score(combo_prob: float, leg1_prob: float, leg2_prob: float) -> tuple[int, ScoreBreakdown]:
    """
    score ∈ [0,100] — four components:
    - prob_base:    combo_prob × 100 (core)
    - legs_bonus:   strength of individual legs beyond 100% combined
    - corr_bonus:   Poisson joint vs statistical independence (± 10 pts)
    - risk_penalty: penalty when combo_prob < 45%
    """
    base = combo_prob * 100
    legs = max(0.0, leg1_prob + leg2_prob - 1.0) * 57

    # Poisson correlation: how much joint prob exceeds independence
    independent = leg1_prob * leg2_prob
    corr_raw = (combo_prob - independent) * 60  # scale to ± ~10 pts typical range
    corr = max(-10, min(10, round(corr_raw)))

    # Risk penalty: combo below 45% is increasingly penalised
    risk_raw = max(0.0, 0.45 - combo_prob) * 40
    risk = -round(risk_raw)

    total = min(100, max(0, round(base + legs + corr + risk)))
    breakdown = ScoreBreakdown(
        prob_base=round(base),
        legs_bonus=round(legs),
        corr_bonus=corr,
        risk_penalty=risk,
        total=total,
    )
    return total, breakdown


def _live_bonus_pts(live: "LivePoissonResult", leg1_label: str, leg2_label: str) -> int:
    """
    Signal-aligned live bonus [−5, +10].
    Higher means live data (intensity, state lock-in, score gap) supports this combo.
    """
    label = f"{leg1_label} {leg2_label}".lower()
    bonus = 0

    # Intensity: shot/xG rate vs expected pace
    avg_i = (live.intensity_home + live.intensity_away) / 2
    is_over_bet = any(x in label for x in ("over", "btts si"))
    is_under_bet = any(x in label for x in ("under", "btts no"))
    if avg_i >= 1.3 and is_over_bet:
        bonus += 4
    elif avg_i <= 0.7 and is_under_bet:
        bonus += 4
    elif avg_i >= 1.3 and is_under_bet:
        bonus -= 2
    elif avg_i <= 0.7 and is_over_bet:
        bonus -= 2

    # State lock-in: fewer minutes remaining → result more certain
    if live.minutes_remaining <= 15:
        bonus += 5
    elif live.minutes_remaining <= 25:
        bonus += 3
    elif live.minutes_remaining <= 40:
        bonus += 1

    # Score gap alignment with DC or outright result bets
    gap = live.home_goals - live.away_goals
    is_home_bet = any(x in label for x in ("1x", "home win", "dc 1x"))
    is_away_bet = any(x in label for x in ("x2", "away win", "dc x2"))
    if gap >= 2 and is_home_bet:
        bonus += 3
    elif gap <= -2 and is_away_bet:
        bonus += 3
    elif abs(gap) == 1 and (is_home_bet or is_away_bet):
        bonus += 1

    return min(10, max(-5, bonus))


def _apply_live_bonus(
    sc: int,
    breakdown: ScoreBreakdown,
    live: "LivePoissonResult",
    leg1_label: str,
    leg2_label: str,
) -> tuple[int, ScoreBreakdown]:
    lb = _live_bonus_pts(live, leg1_label, leg2_label)
    new_total = min(100, max(0, sc + lb))
    return new_total, ScoreBreakdown(
        prob_base=breakdown.prob_base,
        legs_bonus=breakdown.legs_bonus,
        corr_bonus=breakdown.corr_bonus,
        risk_penalty=breakdown.risk_penalty,
        total=new_total,
        live_bonus=lb,
    )


def _decision(score: int, combo_prob: float) -> tuple[str, str, bool]:
    if score >= 65 and combo_prob >= 0.45:
        return "STRONG_BET", "BAJO", True
    if score >= 52 and combo_prob >= 0.38:
        return "MODERATE_BET", "MEDIO", False
    if score >= 38 and combo_prob >= 0.30:
        return "WEAK_BET", "ALTO", False
    return "NO_BET", "MUY ALTO", False


def _make_combo(
    leg1_label: str,
    leg2_label: str,
    p_l1: float,
    p_l2: float,
    combo_p: float,
    disp1: str,
    disp2: str,
    live: "LivePoissonResult | None" = None,
) -> SafeCombo | None:
    """Build a SafeCombo from pre-computed probabilities. Returns None if NO_BET."""
    if p_l1 <= _MIN_LEG_PROB or p_l2 <= _MIN_LEG_PROB:
        return None
    if combo_p < _MIN_COMBO_PROB:
        return None
    sc, breakdown = _score(combo_p, p_l1, p_l2)
    if live is not None:
        sc, breakdown = _apply_live_bonus(sc, breakdown, live, leg1_label, leg2_label)
    dec, risk, rec = _decision(sc, combo_p)
    if dec == "NO_BET":
        return None
    fo = round(1.0 / combo_p, 2)
    return SafeCombo(
        leg1_label=leg1_label,
        leg2_label=leg2_label,
        leg1_prob=round(p_l1, 4),
        leg2_prob=round(p_l2, 4),
        combo_prob=round(combo_p, 4),
        fair_odds=fo,
        market_min_odds=round(fo * _VIG_FACTOR, 2),
        score=sc,
        score_breakdown=breakdown,
        decision=dec,
        risk=risk,
        recommended=rec,
        label1_display=disp1,
        label2_display=disp2,
    )


def _combo_candidates_prematch(model: ModelMarkets, p1: str, p2: str) -> list[tuple]:
    """
    25 candidate 2-leg combos for pre-match.
    Each tuple: (leg1_label, leg2_label, leg1_prob, leg2_prob, condition_fn, disp1, disp2)
    Conditions are over TOTAL final goals (i=home, j=away).
    """
    return [
        # ── Group A: DC + Under (conservative) ───────────────────────────────
        ("DC X2", "Under 2.5",
         model.dc_away_draw, model.under_25,
         lambda i, j: j >= i and i + j <= 2,
         f"X2 (Empate o {p2})", "Under 2.5"),

        ("DC 1X", "Under 2.5",
         model.dc_home_draw, model.under_25,
         lambda i, j: i >= j and i + j <= 2,
         f"1X ({p1} o Empate)", "Under 2.5"),

        ("DC X2", "Under 1.5",
         model.dc_away_draw, model.under_15,
         lambda i, j: j >= i and i + j <= 1,
         f"X2 (Empate o {p2})", "Under 1.5"),

        ("DC 1X", "Under 1.5",
         model.dc_home_draw, model.under_15,
         lambda i, j: i >= j and i + j <= 1,
         f"1X ({p1} o Empate)", "Under 1.5"),

        ("DC 12", "Under 2.5",
         model.dc_home_away, model.under_25,
         lambda i, j: i != j and i + j <= 2,
         f"12 ({p1} o {p2})", "Under 2.5"),

        ("DC 12", "Under 3.5",
         model.dc_home_away, model.under_35,
         lambda i, j: i != j and i + j <= 3,
         f"12 ({p1} o {p2})", "Under 3.5"),

        # ── Group B: DC + Over (value combos) ────────────────────────────────
        ("DC 12", "Over 1.5",
         model.dc_home_away, model.over_15,
         lambda i, j: i != j and i + j > 1,
         f"12 ({p1} o {p2})", "Over 1.5"),

        ("DC X2", "Over 0.5",
         model.dc_away_draw, model.over_05,
         lambda i, j: j >= i and i + j > 0,
         f"X2 (Empate o {p2})", "Over 0.5"),

        ("DC 1X", "Over 0.5",
         model.dc_home_draw, model.over_05,
         lambda i, j: i >= j and i + j > 0,
         f"1X ({p1} o Empate)", "Over 0.5"),

        # ── Group C: DC + BTTS ────────────────────────────────────────────────
        ("DC X2", "BTTS No",
         model.dc_away_draw, model.btts_no,
         lambda i, j: j >= i and (i == 0 or j == 0),
         f"X2 (Empate o {p2})", "BTTS No"),

        ("DC 1X", "BTTS No",
         model.dc_home_draw, model.btts_no,
         lambda i, j: i >= j and (i == 0 or j == 0),
         f"1X ({p1} o Empate)", "BTTS No"),

        ("DC 12", "BTTS Si",
         model.dc_home_away, model.btts_yes,
         lambda i, j: i != j and i >= 1 and j >= 1,
         f"12 ({p1} o {p2})", "BTTS Si"),

        ("DC X2", "BTTS Si",
         model.dc_away_draw, model.btts_yes,
         lambda i, j: j >= i and i >= 1 and j >= 1,
         f"X2 (Empate o {p2})", "BTTS Si"),

        ("DC 1X", "BTTS Si",
         model.dc_home_draw, model.btts_yes,
         lambda i, j: i >= j and i >= 1 and j >= 1,
         f"1X ({p1} o Empate)", "BTTS Si"),

        # ── Group D: Under/Over + BTTS ────────────────────────────────────────
        ("Under 2.5", "BTTS No",
         model.under_25, model.btts_no,
         lambda i, j: i + j <= 2 and (i == 0 or j == 0),
         "Under 2.5", "BTTS No"),

        ("Over 1.5", "BTTS Si",
         model.over_15, model.btts_yes,
         lambda i, j: i + j > 1 and i >= 1 and j >= 1,
         "Over 1.5", "BTTS Si"),

        ("Over 2.5", "BTTS Si",
         model.over_25, model.btts_yes,
         lambda i, j: i + j > 2 and i >= 1 and j >= 1,
         "Over 2.5", "BTTS Si"),

        ("Under 3.5", "BTTS Si",
         model.under_35, model.btts_yes,
         lambda i, j: i + j <= 3 and i >= 1 and j >= 1,
         "Under 3.5", "BTTS Si"),

        ("Over 0.5", "BTTS No",
         model.over_05, model.btts_no,
         lambda i, j: i + j > 0 and (i == 0 or j == 0),
         "Over 0.5", "BTTS No"),

        # ── Group E: Result + Goals ───────────────────────────────────────────
        ("Home Win", "Under 3.5",
         model.home_win, model.under_35,
         lambda i, j: i > j and i + j <= 3,
         f"{p1} gana", "Under 3.5"),

        ("Away Win", "Under 3.5",
         model.away_win, model.under_35,
         lambda i, j: j > i and i + j <= 3,
         f"{p2} gana", "Under 3.5"),

        ("Home Win", "BTTS No",
         model.home_win, model.btts_no,
         lambda i, j: i > j and (i == 0 or j == 0),
         f"{p1} gana", "BTTS No"),

        ("Away Win", "BTTS No",
         model.away_win, model.btts_no,
         lambda i, j: j > i and (i == 0 or j == 0),
         f"{p2} gana", "BTTS No"),

        ("Home Win", "Over 1.5",
         model.home_win, model.over_15,
         lambda i, j: i > j and i + j > 1,
         f"{p1} gana", "Over 1.5"),

        ("Away Win", "Over 1.5",
         model.away_win, model.over_15,
         lambda i, j: j > i and i + j > 1,
         f"{p2} gana", "Over 1.5"),
    ]


def _combo_candidates_live(live: "LivePoissonResult", p1: str, p2: str) -> list[tuple]:
    """
    25 candidate combos for live matches.
    Conditions are over REMAINING goals (di, dj); final score = (g_h+di, g_a+dj).
    """
    g_h = live.home_goals
    g_a = live.away_goals

    return [
        # ── Group A: DC + Under ───────────────────────────────────────────────
        ("DC X2", "Under 2.5",
         live.dc_away_draw, live.under_25,
         lambda di, dj: (g_a + dj) >= (g_h + di) and (g_h + di) + (g_a + dj) <= 2,
         f"X2 (Empate o {p2})", "Under 2.5"),

        ("DC 1X", "Under 2.5",
         live.dc_home_draw, live.under_25,
         lambda di, dj: (g_h + di) >= (g_a + dj) and (g_h + di) + (g_a + dj) <= 2,
         f"1X ({p1} o Empate)", "Under 2.5"),

        ("DC X2", "Under 1.5",
         live.dc_away_draw, live.under_15,
         lambda di, dj: (g_a + dj) >= (g_h + di) and (g_h + di) + (g_a + dj) <= 1,
         f"X2 (Empate o {p2})", "Under 1.5"),

        ("DC 1X", "Under 1.5",
         live.dc_home_draw, live.under_15,
         lambda di, dj: (g_h + di) >= (g_a + dj) and (g_h + di) + (g_a + dj) <= 1,
         f"1X ({p1} o Empate)", "Under 1.5"),

        ("DC 12", "Under 2.5",
         live.dc_home_away, live.under_25,
         lambda di, dj: (g_h + di) != (g_a + dj) and (g_h + di) + (g_a + dj) <= 2,
         f"12 ({p1} o {p2})", "Under 2.5"),

        ("DC 12", "Under 3.5",
         live.dc_home_away, live.under_35,
         lambda di, dj: (g_h + di) != (g_a + dj) and (g_h + di) + (g_a + dj) <= 3,
         f"12 ({p1} o {p2})", "Under 3.5"),

        # ── Group B: DC + Over ────────────────────────────────────────────────
        ("DC 12", "Over 1.5",
         live.dc_home_away, live.over_15,
         lambda di, dj: (g_h + di) != (g_a + dj) and (g_h + di) + (g_a + dj) > 1,
         f"12 ({p1} o {p2})", "Over 1.5"),

        ("DC X2", "Over 0.5",
         live.dc_away_draw, live.over_05,
         lambda di, dj: (g_a + dj) >= (g_h + di) and (g_h + di) + (g_a + dj) > 0,
         f"X2 (Empate o {p2})", "Over 0.5"),

        ("DC 1X", "Over 0.5",
         live.dc_home_draw, live.over_05,
         lambda di, dj: (g_h + di) >= (g_a + dj) and (g_h + di) + (g_a + dj) > 0,
         f"1X ({p1} o Empate)", "Over 0.5"),

        # ── Group C: DC + BTTS ────────────────────────────────────────────────
        ("DC X2", "BTTS No",
         live.dc_away_draw, live.btts_no,
         lambda di, dj: (g_a + dj) >= (g_h + di) and (
             (g_h + di) == 0 or (g_a + dj) == 0
         ),
         f"X2 (Empate o {p2})", "BTTS No"),

        ("DC 1X", "BTTS No",
         live.dc_home_draw, live.btts_no,
         lambda di, dj: (g_h + di) >= (g_a + dj) and (
             (g_h + di) == 0 or (g_a + dj) == 0
         ),
         f"1X ({p1} o Empate)", "BTTS No"),

        ("DC 12", "BTTS Si",
         live.dc_home_away, live.btts_yes,
         lambda di, dj: (g_h + di) != (g_a + dj) and (g_h + di) >= 1 and (g_a + dj) >= 1,
         f"12 ({p1} o {p2})", "BTTS Si"),

        ("DC X2", "BTTS Si",
         live.dc_away_draw, live.btts_yes,
         lambda di, dj: (g_a + dj) >= (g_h + di) and (g_h + di) >= 1 and (g_a + dj) >= 1,
         f"X2 (Empate o {p2})", "BTTS Si"),

        ("DC 1X", "BTTS Si",
         live.dc_home_draw, live.btts_yes,
         lambda di, dj: (g_h + di) >= (g_a + dj) and (g_h + di) >= 1 and (g_a + dj) >= 1,
         f"1X ({p1} o Empate)", "BTTS Si"),

        # ── Group D: Under/Over + BTTS ────────────────────────────────────────
        ("Under 2.5", "BTTS No",
         live.under_25, live.btts_no,
         lambda di, dj: (g_h + di) + (g_a + dj) <= 2 and (
             (g_h + di) == 0 or (g_a + dj) == 0
         ),
         "Under 2.5", "BTTS No"),

        ("Over 1.5", "BTTS Si",
         live.over_15, live.btts_yes,
         lambda di, dj: (g_h + di) + (g_a + dj) > 1 and (g_h + di) >= 1 and (g_a + dj) >= 1,
         "Over 1.5", "BTTS Si"),

        ("Over 2.5", "BTTS Si",
         live.over_25, live.btts_yes,
         lambda di, dj: (g_h + di) + (g_a + dj) > 2 and (g_h + di) >= 1 and (g_a + dj) >= 1,
         "Over 2.5", "BTTS Si"),

        ("Under 3.5", "BTTS Si",
         live.under_35, live.btts_yes,
         lambda di, dj: (g_h + di) + (g_a + dj) <= 3 and (g_h + di) >= 1 and (g_a + dj) >= 1,
         "Under 3.5", "BTTS Si"),

        ("Over 0.5", "BTTS No",
         live.over_05, live.btts_no,
         lambda di, dj: (g_h + di) + (g_a + dj) > 0 and (
             (g_h + di) == 0 or (g_a + dj) == 0
         ),
         "Over 0.5", "BTTS No"),

        # ── Group E: Result + Goals ───────────────────────────────────────────
        ("Home Win", "Under 3.5",
         live.home_win, live.under_35,
         lambda di, dj: (g_h + di) > (g_a + dj) and (g_h + di) + (g_a + dj) <= 3,
         f"{p1} gana", "Under 3.5"),

        ("Away Win", "Under 3.5",
         live.away_win, live.under_35,
         lambda di, dj: (g_a + dj) > (g_h + di) and (g_h + di) + (g_a + dj) <= 3,
         f"{p2} gana", "Under 3.5"),

        ("Home Win", "BTTS No",
         live.home_win, live.btts_no,
         lambda di, dj: (g_h + di) > (g_a + dj) and (
             (g_h + di) == 0 or (g_a + dj) == 0
         ),
         f"{p1} gana", "BTTS No"),

        ("Away Win", "BTTS No",
         live.away_win, live.btts_no,
         lambda di, dj: (g_a + dj) > (g_h + di) and (
             (g_h + di) == 0 or (g_a + dj) == 0
         ),
         f"{p2} gana", "BTTS No"),

        ("Home Win", "Over 1.5",
         live.home_win, live.over_15,
         lambda di, dj: (g_h + di) > (g_a + dj) and (g_h + di) + (g_a + dj) > 1,
         f"{p1} gana", "Over 1.5"),

        ("Away Win", "Over 1.5",
         live.away_win, live.over_15,
         lambda di, dj: (g_a + dj) > (g_h + di) and (g_h + di) + (g_a + dj) > 1,
         f"{p2} gana", "Over 1.5"),
    ]


def build_safe_combinations(
    model: ModelMarkets,
    team1: str = "",
    team2: str = "",
) -> list[SafeCombo]:
    """
    Generate ranked intra-match 2-leg combinations using exact Poisson joint probabilities.
    Returns up to 8 combos sorted by score descending; excludes NO_BET and prob < 30%.

    NOTE: For live matches (current score != 0-0), use build_live_combinations() instead.
    This function assumes lambdas represent full-game expected goals and conditions are
    evaluated over total final goals. Calling it with remaining-game lambdas will produce
    incorrect probabilities because conditions do not account for goals already scored.
    """
    try:
        matrix = _build_matrix(model.lambda_home, model.lambda_away)
    except Exception:
        return []

    p1 = team1.split()[0] if team1 else "Local"
    p2 = team2.split()[0] if team2 else "Visit."
    candidates = _combo_candidates_prematch(model, p1, p2)

    results: list[SafeCombo] = []
    for leg1_label, leg2_label, p_l1, p_l2, cond, disp1, disp2 in candidates:
        combo_p = _joint(matrix, cond)
        item = _make_combo(leg1_label, leg2_label, p_l1, p_l2, combo_p, disp1, disp2)
        if item:
            results.append(item)

    results.sort(key=lambda x: (-x.score, -x.combo_prob))
    return results[:_MAX_RESULTS]


def build_live_combinations(
    live: "LivePoissonResult",
    team1: str = "",
    team2: str = "",
) -> list[SafeCombo]:
    """
    Live-aware combo engine.

    Recomputes all 25 2-leg combinations conditioned on the current score
    (home_goals, away_goals) using the remaining-goals Poisson matrix.
    Conditions are shifted: i,j = REMAINING goals, so final score = (g_h+i, g_a+j).
    Scores are enriched with live signal bonus (intensity, state lock-in, score gap).
    """
    from apps.worker.ml.poisson_live import build_live_score_matrix

    try:
        matrix = build_live_score_matrix(
            live.lambda_home_remaining,
            live.lambda_away_remaining,
        )
    except Exception:
        return []

    p1 = team1.split()[0] if team1 else "Local"
    p2 = team2.split()[0] if team2 else "Visit."
    candidates = _combo_candidates_live(live, p1, p2)

    results: list[SafeCombo] = []
    for leg1_label, leg2_label, p_l1, p_l2, cond, disp1, disp2 in candidates:
        combo_p = _joint(matrix, cond)
        item = _make_combo(leg1_label, leg2_label, p_l1, p_l2, combo_p, disp1, disp2, live=live)
        if item:
            results.append(item)

    results.sort(key=lambda x: (-x.score, -x.combo_prob))
    return results[:_MAX_RESULTS]
