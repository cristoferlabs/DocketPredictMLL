"""
Live calibration layer — MODEL → tournament → α → shrink → EV.

Invariante: P_statistical inmutable en blend_meta; P_calibrated en campos públicos.
Mercado entra UNA sola vez (α). Shrink usa solo P_stat + P_cal (guard rail).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets
from apps.shared.config import Settings, get_settings
from apps.worker.ml.model_combiner import Probabilities1X2, apply_market_calibration_layer
from apps.worker.ml.odds_math import fair_h2h_market, fair_totals_market

OUTCOMES_1X2 = ("home_win", "draw", "away_win")


def _lambda_from_under25(under_25: float, lam_init: float) -> float:
    """Solve P(Poisson(λ) ≤ 2) = under_25 via Newton's method."""
    if under_25 >= 0.995:
        return 0.05
    if under_25 <= 0.005:
        return 7.0
    lam = max(0.1, min(7.0, lam_init))
    for _ in range(15):
        e_lam = math.exp(-lam)
        cdf2 = e_lam * (1.0 + lam + lam * lam * 0.5)
        # f'(λ) = -e^(-λ) * λ²/2  (always ≤ 0)
        deriv = -e_lam * lam * lam * 0.5
        if abs(deriv) < 1e-12:
            break
        lam = max(0.05, min(8.0, lam - (cdf2 - under_25) / deriv))
    return round(lam, 6)


def _recompute_btts_from_lambda(lam_home: float, lam_away: float) -> tuple[float, float]:
    """BTTS No = P(home=0) + P(away=0) - P(both=0)."""
    p_h0 = math.exp(-lam_home)
    p_a0 = math.exp(-lam_away)
    btts_no = min(0.99, max(0.01, p_h0 + p_a0 - math.exp(-(lam_home + lam_away))))
    return round(btts_no, 4), round(1.0 - btts_no, 4)
AlphaRegime = Literal["aligned", "moderate", "high", "extreme"]
PreAlphaBucket = Literal["favorite_strong", "favorite_medium", "balanced", "underdog"]


@dataclass
class LiveCalibrationContext:
    competition: str = "fifa_world_cup"
    max_divergence_pp: float = 0.0
    data_quality_pct: float = 100.0
    hist_played: int = 20
    n_books: int = 0
    round_name: str = ""
    mus: float | None = None


@dataclass
class LiveCalibrationResult:
    calibrated: ModelMarkets
    statistical: dict[str, float]
    alpha: float
    shrink_applied: bool
    tournament_applied: bool
    meta: dict[str, Any] = field(default_factory=dict)


def _model_1x2_dict(model: ModelMarkets) -> dict[str, float]:
    return {
        "home_win": model.home_win,
        "draw": model.draw,
        "away_win": model.away_win,
    }


def _totals_dict(model: ModelMarkets) -> dict[str, float]:
    return {"over_25": model.over_25, "under_25": model.under_25}


def _max_divergence_pp(a: dict[str, float], b: dict[str, float], keys: tuple[str, ...]) -> float:
    return max(abs(a.get(k, 0.0) - b.get(k, 0.0)) * 100.0 for k in keys)


def _is_knockout(round_name: str) -> bool:
    rn = (round_name or "").lower()
    return any(x in rn for x in ("round of", "quarter", "semi", "final", "octavos", "cuartos"))


def resolve_alpha_regime(
    delta_pp: float,
    settings: Settings,
    *,
    is_wc: bool,
) -> tuple[float, AlphaRegime]:
    """
    α piecewise por Δ(P_stat, P_market) — calibración tipo desk.

    Δ < 10  → low     (0.25–0.35 WC)
    10–20   → medium  (0.45–0.60)
    20–30   → high    (0.65–0.75)
    ≥ 30    → max     (0.75–0.80)
    """
    t1, t2, t3 = settings.cal_alpha_regime_t1, settings.cal_alpha_regime_t2, settings.cal_alpha_regime_t3
    if is_wc:
        bands = (
            (t1, settings.cal_alpha_regime_low, "aligned"),
            (t2, settings.cal_alpha_regime_medium, "moderate"),
            (t3, settings.cal_alpha_regime_high, "high"),
        )
        ceiling = settings.cal_alpha_regime_max
    else:
        bands = (
            (t1, settings.cal_alpha_normal_low, "aligned"),
            (t2, settings.cal_alpha_normal_medium, "moderate"),
            (t3, settings.cal_alpha_normal_high, "high"),
        )
        ceiling = settings.cal_alpha_normal_max

    if delta_pp < t1:
        return bands[0][1], "aligned"  # type: ignore[return-value]
    if delta_pp < t2:
        return bands[1][1], "moderate"  # type: ignore[return-value]
    if delta_pp < t3:
        return bands[2][1], "high"  # type: ignore[return-value]
    return ceiling, "extreme"


def compute_dynamic_alpha(ctx: LiveCalibrationContext, settings: Settings) -> float:
    """α por régimen — no lineal; mercado pesa según extremo del mismatch."""
    is_wc = ctx.competition == "fifa_world_cup"
    delta = ctx.max_divergence_pp
    alpha, _regime = resolve_alpha_regime(delta, settings, is_wc=is_wc)

    floor = settings.cal_alpha_regime_low if is_wc else settings.cal_alpha_normal_low
    ceiling = settings.cal_alpha_regime_max if is_wc else settings.cal_alpha_normal_max

    if _is_knockout(ctx.round_name):
        alpha = min(ceiling, alpha + settings.cal_alpha_knockout_bump)
    if 0 < ctx.n_books < 3:
        alpha = max(floor, alpha - settings.cal_alpha_thin_books_penalty)
    if ctx.mus is not None and ctx.mus < 0.85:
        alpha = min(ceiling, alpha + settings.cal_alpha_low_mus_bump)
    if ctx.data_quality_pct < 60.0 and delta >= settings.cal_alpha_regime_t2:
        alpha = min(ceiling, alpha + settings.cal_alpha_low_mus_bump)

    return round(min(ceiling, max(floor, alpha)), 4)


def compute_alpha_with_regime(
    ctx: LiveCalibrationContext,
    settings: Settings,
) -> tuple[float, AlphaRegime]:
    """α + etiqueta de régimen para telemetría SHARP."""
    is_wc = ctx.competition == "fifa_world_cup"
    _, regime = resolve_alpha_regime(ctx.max_divergence_pp, settings, is_wc=is_wc)
    return compute_dynamic_alpha(ctx, settings), regime


def cap_alpha_for_alignment(
    alpha: float,
    divergence_cal_pp: float,
    *,
    settings: Settings,
) -> float:
    """Anti-overfit: si ya alineado con mercado, no subir α."""
    capped = alpha
    if divergence_cal_pp < 5.0:
        capped = min(capped, settings.cal_alpha_aligned_cap)
    if capped > settings.cal_alpha_overfit_cap and divergence_cal_pp < 5.0:
        capped = settings.cal_alpha_overfit_cap
    return round(capped, 4)


def _classify_pre_alpha_bucket(prob: float, peak: float) -> PreAlphaBucket:
    if prob < 0.40:
        return "underdog"
    if prob >= peak - 1e-9 and peak >= 0.62:
        return "favorite_strong"
    if prob >= peak - 1e-9 and peak >= 0.48:
        return "favorite_medium"
    if prob >= 0.55:
        return "favorite_medium"
    return "balanced"


def apply_pre_alpha_bucket_calibration(
    probs: dict[str, float],
    *,
    settings: Settings,
    market_fair: dict[str, float] | None = None,
    bucket_config: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
  P_stat → bucket isotonic/slope (pre-α).

  Corrige cola estructural antes de mezclar con mercado.
  """
    if not settings.pre_alpha_bucket_enabled:
        return probs, {"pre_alpha_bucket": False}

    from apps.worker.ml.calibration import apply_bucket_1x2, load_fitted_calibration_factors

    h0, d0, a0 = probs["home_win"], probs["draw"], probs["away_win"]
    cfg = bucket_config
    if cfg is None:
        fitted = load_fitted_calibration_factors()
        cfg = (fitted or {}).get("1X2_buckets")

    h, d, a = apply_bucket_1x2(h0, d0, a0, cfg)
    peak = max(h, a)
    buckets_hit: list[str] = []

    h_bucket = _classify_pre_alpha_bucket(h, peak)
    a_bucket = _classify_pre_alpha_bucket(a, peak)
    buckets_hit.extend(sorted({h_bucket, a_bucket}))

    if peak >= settings.pre_alpha_favorite_strong_min:
        factor = settings.pre_alpha_favorite_strong_factor
        if h >= a:
            h = min(0.88, h * factor)
        else:
            a = min(0.88, a * factor)
        if "favorite_strong" not in buckets_hit:
            buckets_hit.append("favorite_strong")

    if market_fair:
        comp_pp = settings.pre_alpha_market_compression_pp
        for key, val, mkt_key in (
            ("home_win", h, "home_win"),
            ("away_win", a, "away_win"),
        ):
            mkt_p = market_fair.get(mkt_key)
            if mkt_p is None:
                continue
            gap_pp = (mkt_p - val) * 100.0
            if gap_pp > comp_pp and val >= 0.45:
                gain = min(0.10, gap_pp / 100.0 * settings.pre_alpha_market_compression_gain)
                if key == "home_win":
                    h = min(0.88, h + gain)
                else:
                    a = min(0.88, a + gain)
                buckets_hit.append("market_compression_lift")
            elif (val - mkt_p) * 100.0 > settings.pre_alpha_underdog_inflation_pp and val < 0.42:
                damp = settings.pre_alpha_underdog_dampen
                if key == "home_win":
                    h *= damp
                else:
                    a *= damp
                buckets_hit.append("underdog_inflation_dampen")

    total = h + d + a
    if total <= 0:
        return probs, {"pre_alpha_bucket": False}
    out = {"home_win": h / total, "draw": d / total, "away_win": a / total}
    return out, {
        "pre_alpha_bucket": True,
        "pre_alpha_buckets": sorted(set(buckets_hit)),
        "pre_alpha_peak": round(peak, 4),
    }


def apply_tournament_factors(
    probs: dict[str, float],
    totals: dict[str, float] | None,
    *,
    competition: str,
    settings: Settings,
) -> tuple[dict[str, float], dict[str, float] | None, bool]:
    """Factores estructurales WC — antes del shrink mercado."""
    if competition != "fifa_world_cup":
        return probs, totals, False

    h, d, a = probs["home_win"], probs["draw"], probs["away_win"]
    d *= 1.0 + settings.tournament_draw_boost
    rest = h + a
    if rest > 0:
        scale = (1.0 - d) / rest
        h, a = h * scale, a * scale
    out_1x2 = {"home_win": h, "draw": d, "away_win": a}

    out_totals = totals
    if totals:
        ou = totals.get("over_25", 0.5)
        un = totals.get("under_25", 0.5)
        un *= 1.0 + settings.tournament_under_boost
        ou *= 1.0 - settings.tournament_over_penalty
        total_ou = ou + un
        if total_ou > 0:
            out_totals = {
                "over_25": ou / total_ou,
                "under_25": un / total_ou,
            }

    return out_1x2, out_totals, True


def apply_underdog_shrink(
    p_stat: dict[str, float],
    p_cal: dict[str, float],
    *,
    gap_pp_threshold: float,
    stat_weight: float,
    mismatch_high: bool,
    divergence_cal_pp: float,
) -> tuple[dict[str, float], bool]:
    """
    Guard rail — solo si Δ_cal extrema Y mismatch alto.
    Mezcla P_stat + P_cal (sin re-usar mercado).
    """
    if not mismatch_high or divergence_cal_pp <= gap_pp_threshold:
        return p_cal, False

    cal_weight = 1.0 - stat_weight
    adjusted: dict[str, float] = {}
    applied = False
    for key in OUTCOMES_1X2:
        gap_pp = abs(p_stat[key] - p_cal[key]) * 100.0
        if gap_pp > gap_pp_threshold:
            adjusted[key] = stat_weight * p_stat[key] + cal_weight * p_cal[key]
            applied = True
        else:
            adjusted[key] = p_cal[key]

    if not applied:
        return p_cal, False

    total = sum(adjusted.values())
    if total <= 0:
        return p_cal, False
    return {k: adjusted[k] / total for k in OUTCOMES_1X2}, True


def _market_fair_from_event(odds_event: dict | None) -> tuple[dict[str, float] | None, dict[str, float] | None, int]:
    if not odds_event:
        return None, None, 0
    n_books = len(odds_event.get("bookmakers", []))
    h2h = fair_h2h_market(odds_event)
    fair_1x2: dict[str, float] = {}
    for key in ("home", "draw", "away"):
        fm = h2h.get(key, {})
        if fm.get("fair_prob"):
            fair_1x2[{"home": "home_win", "draw": "draw", "away": "away_win"}[key]] = fm["fair_prob"]

    totals_raw = fair_totals_market(odds_event, 2.5)
    fair_totals: dict[str, float] = {}
    if totals_raw.get("over", {}).get("fair_prob"):
        fair_totals["over_25"] = totals_raw["over"]["fair_prob"]
    if totals_raw.get("under", {}).get("fair_prob"):
        fair_totals["under_25"] = totals_raw["under"]["fair_prob"]

    if len(fair_1x2) < 3:
        return None, fair_totals or None, n_books
    return fair_1x2, fair_totals or None, n_books


def _overfit_warning(divergence_stat_pp: float, divergence_cal_pp: float, settings: Settings) -> bool:
    if divergence_stat_pp < 8.0:
        return False
    ratio = divergence_cal_pp / divergence_stat_pp if divergence_stat_pp > 0 else 1.0
    return ratio < settings.overfit_warning_ratio


def apply_live_calibration(
    model: ModelMarkets,
    market_fair_1x2: dict[str, float] | None,
    *,
    market_fair_totals: dict[str, float] | None = None,
    context: LiveCalibrationContext,
    settings: Settings | None = None,
) -> LiveCalibrationResult:
    """
    Orden: P_stat → pre-α bucket → tournament → α régimen → shrink → P_cal.
    """
    settings = settings or get_settings()
    p_stat = _model_1x2_dict(model)
    statistical = copy.deepcopy(p_stat)
    totals_stat = _totals_dict(model)

    if settings.live_calibration_enabled:
        p_work, pre_alpha_meta = apply_pre_alpha_bucket_calibration(
            p_stat,
            settings=settings,
            market_fair=market_fair_1x2,
        )
        p_work, totals_work, tournament_applied = apply_tournament_factors(
            p_work,
            totals_stat,
            competition=context.competition,
            settings=settings,
        )
    else:
        p_work = copy.deepcopy(p_stat)
        totals_work = totals_stat
        tournament_applied = False
        pre_alpha_meta = {"pre_alpha_bucket": False}

    alpha = 0.0
    alpha_regime: AlphaRegime | str = "no_market"
    shrink_applied = False
    p_cal = copy.deepcopy(p_work)
    divergence_stat_pp = 0.0
    divergence_cal_pp = 0.0
    overfit_warn = False

    if market_fair_1x2 and settings.live_calibration_enabled:
        divergence_stat_pp = _max_divergence_pp(p_work, market_fair_1x2, OUTCOMES_1X2)
        alpha_raw, alpha_regime = compute_alpha_with_regime(context, settings)
        alpha = alpha_raw
        p_alpha = apply_market_calibration_layer(
            Probabilities1X2.from_mapping(p_work),
            market_fair_1x2,
            alpha=alpha,
        ).as_dict()
        divergence_cal_pp = _max_divergence_pp(p_alpha, market_fair_1x2, OUTCOMES_1X2)
        alpha_capped = cap_alpha_for_alignment(alpha, divergence_cal_pp, settings=settings)
        if alpha_capped != alpha:
            alpha = alpha_capped
            p_alpha = apply_market_calibration_layer(
                Probabilities1X2.from_mapping(p_work),
                market_fair_1x2,
                alpha=alpha,
            ).as_dict()
            divergence_cal_pp = _max_divergence_pp(p_alpha, market_fair_1x2, OUTCOMES_1X2)

        mismatch_high = (
            context.max_divergence_pp >= settings.underdog_shrink_mismatch_pp
            or context.data_quality_pct < 60.0
        )
        p_cal, shrink_applied = apply_underdog_shrink(
            p_work,
            p_alpha,
            gap_pp_threshold=settings.underdog_shrink_gap_pp,
            stat_weight=settings.underdog_shrink_stat_weight,
            mismatch_high=mismatch_high,
            divergence_cal_pp=divergence_cal_pp,
        )
        overfit_warn = _overfit_warning(divergence_stat_pp, divergence_cal_pp, settings)
    elif not settings.live_calibration_enabled:
        p_cal = p_work
        alpha_regime = "disabled"
    else:
        alpha_regime = "no_market"

    if totals_work and market_fair_totals and settings.live_calibration_enabled:
        ou_keys = ("over_25", "under_25")
        if all(k in market_fair_totals for k in ou_keys):
            t_alpha = min(alpha, settings.cal_alpha_aligned_cap) if alpha > 0 else 0.0
            if t_alpha > 0:
                over = totals_work["over_25"] * (1 - t_alpha) + market_fair_totals["over_25"] * t_alpha
                under = totals_work["under_25"] * (1 - t_alpha) + market_fair_totals["under_25"] * t_alpha
                total_ou = over + under
                if total_ou > 0:
                    totals_work = {"over_25": over / total_ou, "under_25": under / total_ou}

    # --- Propagate calibrated Under 2.5 → BTTS + lambdas (consistency fix) ---
    # When the market calibrates Under 2.5, BTTS and the Poisson matrix used by
    # safe_combo_engine must also update — otherwise the combo stays frozen at
    # pre-match values even as market odds shift.
    btts_no = model.btts_no
    btts_yes = model.btts_yes
    lam_home_cal = model.lambda_home
    lam_away_cal = model.lambda_away

    lam_total_orig = model.lambda_home + model.lambda_away
    under_orig = totals_stat.get("under_25", model.under_25)
    under_cal = (totals_work or {}).get("under_25", under_orig)

    if lam_total_orig > 0.1 and abs(under_cal - under_orig) > 0.003:
        lam_total_cal = _lambda_from_under25(under_cal, lam_total_orig)
        ratio_h = model.lambda_home / lam_total_orig
        lam_home_cal = round(max(0.1, lam_total_cal * ratio_h), 4)
        lam_away_cal = round(max(0.1, lam_total_cal * (1.0 - ratio_h)), 4)
        # Additive delta: anchor to calibrated BTTS level, apply only the Poisson-derived change.
        # Avoids overwriting calibration factors while keeping directionality correct.
        btts_no_raw_orig, _ = _recompute_btts_from_lambda(model.lambda_home, model.lambda_away)
        btts_no_raw_new, _ = _recompute_btts_from_lambda(lam_home_cal, lam_away_cal)
        delta = btts_no_raw_new - btts_no_raw_orig
        btts_no = round(min(0.99, max(0.01, model.btts_no + delta)), 4)
        btts_yes = round(1.0 - btts_no, 4)

    # DC fields from calibrated 1X2 (previously dropped to 0.0 in calibrated output)
    dc_home_draw = round(p_cal["home_win"] + p_cal["draw"], 4)
    dc_away_draw = round(p_cal["draw"] + p_cal["away_win"], 4)
    dc_home_away = round(p_cal["home_win"] + p_cal["away_win"], 4)

    meta: dict[str, Any] = {
        "engine": "live_calibration_v3_pre_alpha",
        "alpha": alpha,
        "alpha_regime": alpha_regime,
        "shrink_applied": shrink_applied,
        "tournament_applied": tournament_applied,
        "divergence_stat_pp": round(divergence_stat_pp, 2),
        "divergence_cal_pp": round(divergence_cal_pp, 2),
        "overfit_warning": overfit_warn,
        "competition": context.competition,
        "order": "stat→pre_alpha_bucket→tournament→alpha_regime→shrink",
        **pre_alpha_meta,
    }
    if overfit_warn:
        meta["warning"] = "overfitting_to_market"

    blend_meta = dict(model.blend_meta or {})
    blend_meta["statistical"] = statistical
    blend_meta["calibration"] = meta
    blend_meta["divergence_stat_pp"] = meta["divergence_stat_pp"]
    blend_meta["divergence_cal_pp"] = meta["divergence_cal_pp"]

    calibrated = ModelMarkets(
        home_win=round(p_cal["home_win"], 4),
        draw=round(p_cal["draw"], 4),
        away_win=round(p_cal["away_win"], 4),
        over_25=round(totals_work["over_25"], 4),
        under_25=round(totals_work["under_25"], 4),
        btts_yes=btts_yes,
        btts_no=btts_no,
        lambda_home=lam_home_cal,
        lambda_away=lam_away_cal,
        confidence=model.confidence,
        blend_meta=blend_meta,
        # Extended totals preserved from Poisson model (small λ changes don't warrant full recompute)
        over_05=model.over_05,
        over_15=model.over_15,
        over_35=model.over_35,
        over_45=model.over_45,
        under_05=model.under_05,
        under_15=model.under_15,
        under_35=model.under_35,
        under_45=model.under_45,
        # DC recomputed from calibrated 1X2 (previously defaulted to 0.0)
        dc_home_draw=dc_home_draw,
        dc_away_draw=dc_away_draw,
        dc_home_away=dc_home_away,
    )

    return LiveCalibrationResult(
        calibrated=calibrated,
        statistical=statistical,
        alpha=alpha,
        shrink_applied=shrink_applied,
        tournament_applied=tournament_applied,
        meta=meta,
    )


def calibrate_analysis_model(
    analysis: MatchAnalysis,
    odds_event: dict | None,
    *,
    data_quality_pct: float = 100.0,
    hist_played: int = 20,
    mus: float | None = None,
    settings: Settings | None = None,
) -> LiveCalibrationResult | None:
    """Cableado orquestador — aplica calibración in-place sobre analysis.model."""
    if not analysis.model:
        return None
    settings = settings or get_settings()
    fair_1x2, fair_totals, n_books = _market_fair_from_event(odds_event)

    p_stat = _model_1x2_dict(analysis.model)
    div_stat_pp = _max_divergence_pp(p_stat, fair_1x2, OUTCOMES_1X2) if fair_1x2 else 0.0

    ctx = LiveCalibrationContext(
        competition="fifa_world_cup",
        max_divergence_pp=div_stat_pp,
        data_quality_pct=data_quality_pct,
        hist_played=hist_played,
        n_books=n_books,
        round_name=analysis.ronda or "",
        mus=mus,
    )
    result = apply_live_calibration(
        analysis.model,
        fair_1x2,
        market_fair_totals=fair_totals,
        context=ctx,
        settings=settings,
    )
    analysis.model = result.calibrated
    return result
