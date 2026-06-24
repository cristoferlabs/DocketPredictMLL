"""Walk-forward backtesting for World Cup model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from apps.worker.ml.calibration import (
    IsotonicCalibrator,
    brier_score_multiclass,
    expected_calibration_error,
    reliability_bins,
)
from apps.worker.ml.wc_historical import (
    actual_outcomes,
    extract_finished_matches,
    iter_walk_forward_windows,
    predict_match_historical,
)


@dataclass
class BacktestMetrics:
    mode: str
    sample_size: int
    brier_1x2: float
    ece_over: float
    ece_under: float
    ece_btts: float
    hit_rate_1x2: float
    roi_sim: float | None = None
    roi_ci_low: float | None = None
    roi_ci_high: float | None = None
    windows: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "sample_size": self.sample_size,
            "brier_1x2": self.brier_1x2,
            "ece_over": self.ece_over,
            "ece_under": self.ece_under,
            "ece_btts": self.ece_btts,
            "hit_rate_1x2": self.hit_rate_1x2,
            "roi_sim": self.roi_sim,
            "roi_ci_low": self.roi_ci_low,
            "roi_ci_high": self.roi_ci_high,
            "windows": self.windows,
            "details": self.details,
        }


def _collect_predictions(
    archives: dict[int, dict],
    matches: list,
    calibrators: dict[str, IsotonicCalibrator] | None = None,
) -> dict[str, list]:
    """Run model on matches; optionally apply pre-fitted calibrators."""
    data: dict[str, list] = {
        "probs_1x2": [],
        "labels_1x2": [],
        "p_over": [],
        "y_over": [],
        "p_under": [],
        "y_under": [],
        "p_btts": [],
        "y_btts": [],
        "hit_correct": 0,
        "hit_total": 0,
    }
    for m in matches:
        probs = predict_match_historical(m, archives)
        if calibrators:
            for key, cal in calibrators.items():
                if key in probs:
                    probs[key] = cal.transform(probs[key])
            # Renormalize 1X2 after per-outcome calibration
            t = probs["home_win"] + probs["draw"] + probs["away_win"]
            if t > 0:
                probs["home_win"] /= t
                probs["draw"] /= t
                probs["away_win"] /= t

        actual = actual_outcomes(m)
        data["probs_1x2"].append([probs["home_win"], probs["draw"], probs["away_win"]])
        data["labels_1x2"].append(actual["label_1x2"])
        data["p_over"].append(probs["over_25"])
        data["y_over"].append(actual["over_25"])
        data["p_under"].append(probs["under_25"])
        data["y_under"].append(actual["under_25"])
        data["p_btts"].append(probs["btts_yes"])
        data["y_btts"].append(actual["btts_yes"])

        best_idx = int(np.argmax([probs["home_win"], probs["draw"], probs["away_win"]]))
        data["hit_correct"] += int(best_idx == actual["label_1x2"])
        data["hit_total"] += 1

    return data


def _fit_calibrators(
    archives: dict[int, dict],
    train_matches: list,
) -> dict[str, IsotonicCalibrator]:
    """Fit isotonic calibrators on train window only."""
    buckets: dict[str, tuple[list[float], list[int]]] = {
        "home_win": ([], []),
        "draw": ([], []),
        "away_win": ([], []),
        "over_25": ([], []),
        "under_25": ([], []),
        "btts_yes": ([], []),
    }
    for m in train_matches:
        probs = predict_match_historical(m, archives)
        actual = actual_outcomes(m)
        for key in buckets:
            buckets[key][0].append(probs[key])
            buckets[key][1].append(actual[key])

    calibrators: dict[str, IsotonicCalibrator] = {}
    for market, (ps, ys) in buckets.items():
        cal = IsotonicCalibrator(market)
        cal.fit(ps, ys)
        calibrators[market] = cal
    return calibrators


def _max_drawdown(pnl_series: list[float]) -> float:
    """Peak-to-trough drawdown on cumulative PnL."""
    if not pnl_series:
        return 0.0
    peak = 0.0
    cum = 0.0
    max_dd = 0.0
    for p in pnl_series:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return round(max_dd, 4)


def _load_historical_odds_index(db) -> dict[str, dict[str, float]]:
    """Best-effort index team_pair -> h2h raw odds from latest odds-api ingestions."""
    index: dict[str, dict[str, float]] = {}
    if db is None:
        return index
    try:
        rows = (
            db.schema("ops")
            .table("raw_ingestions")
            .select("payload")
            .eq("source", "odds-api")
            .order("ingested_at", desc=True)
            .limit(50)
            .execute()
        )
        from apps.worker.ml.odds_math import fair_h2h_market

        for row in rows.data or []:
            payload = row.get("payload") or {}
            events = payload if isinstance(payload, list) else payload.get("data", [])
            if not isinstance(events, list):
                continue
            for ev in events:
                home = (ev.get("home_team") or "").lower()
                away = (ev.get("away_team") or "").lower()
                if not home or not away:
                    continue
                fair = fair_h2h_market(ev)
                key = f"{home}|{away}"
                if key not in index:
                    index[key] = {
                        "home_win": fair.get("home", {}).get("raw_odds", 0),
                        "draw": fair.get("draw", {}).get("raw_odds", 0),
                        "away_win": fair.get("away", {}).get("raw_odds", 0),
                    }
    except Exception:
        pass
    return index


def simulate_roi_flat_ev(
    archives: dict[int, dict],
    matches: list,
    *,
    min_edge: float = 0.03,
    flat_stake: float = 1.0,
    db=None,
) -> tuple[float | None, float | None, dict[str, Any]]:
    """
    Flat-stake simulation on +EV fair 1X2 picks when book odds exist in raw_ingestions.
    Returns (roi, max_drawdown, details). roi=None if no odds matched.
    """
    from apps.shared.config import get_settings
    from apps.worker.ml.odds_math import expected_value_fair

    odds_index = _load_historical_odds_index(db)
    if not odds_index:
        return None, None, {"mode": "model_only", "reason": "no historical odds in raw_ingestions"}

    settings = get_settings()
    min_edge = min_edge or settings.ev_min_edge_fair
    pnl: list[float] = []
    bets = 0
    wins = 0

    for m in matches:
        probs = predict_match_historical(m, archives)
        actual = actual_outcomes(m)
        key = f"{m.team1.lower()}|{m.team2.lower()}"
        odds = odds_index.get(key)
        if not odds:
            continue

        best_outcome = max(
            [("home_win", probs["home_win"]), ("draw", probs["draw"]), ("away_win", probs["away_win"])],
            key=lambda x: x[1],
        )
        market, prob = best_outcome
        book_odds = odds.get(market, 0)
        if book_odds <= 1:
            continue
        ev = expected_value_fair(prob, book_odds)
        if ev < min_edge:
            continue

        bets += 1
        label_map = {"home_win": 0, "draw": 1, "away_win": 2}
        won = label_map[market] == actual["label_1x2"]
        pnl.append(flat_stake * (book_odds - 1) if won else -flat_stake)
        if won:
            wins += 1

    if bets == 0:
        return None, None, {"mode": "no_bets", "odds_index_size": len(odds_index)}

    total_staked = bets * flat_stake
    roi = round(sum(pnl) / total_staked, 4)
    return roi, _max_drawdown(pnl), {"bets": bets, "wins": wins, "hit_rate": round(wins / bets, 4)}


def _aggregate_metrics(
    all_probs_1x2: list,
    all_labels: list,
    p_over: list,
    y_over: list,
    p_under: list,
    y_under: list,
    p_btts: list,
    y_btts: list,
    hit_correct: int,
    hit_total: int,
    mode: str,
    windows: int,
    roi_sim: float | None = None,
    max_drawdown: float | None = None,
    roi_details: dict[str, Any] | None = None,
) -> BacktestMetrics:
    n = len(all_labels)
    if n == 0:
        return BacktestMetrics(mode=mode, sample_size=0, brier_1x2=0, ece_over=0, ece_under=0, ece_btts=0, hit_rate_1x2=0, windows=windows)

    return BacktestMetrics(
        mode=mode,
        sample_size=n,
        brier_1x2=brier_score_multiclass(all_probs_1x2, all_labels),
        ece_over=expected_calibration_error(p_over, y_over),
        ece_under=expected_calibration_error(p_under, y_under),
        ece_btts=expected_calibration_error(p_btts, y_btts),
        hit_rate_1x2=round(hit_correct / hit_total, 4) if hit_total else 0.0,
        roi_sim=roi_sim,
        windows=windows,
        details={
            "reliability_over": reliability_bins(p_over, y_over, n_bins=5),
            "min_sample_ok": n >= 30,
            "max_drawdown": max_drawdown,
            "roi_details": roi_details or {},
        },
    )


def run_walk_forward_backtest(
    archives: dict[int, dict],
    *,
    train_size: int = 40,
    test_size: int = 20,
    years: list[int] | None = None,
    apply_calibration: bool = True,
    db=None,
) -> BacktestMetrics:
    """
    Walk-forward by match count. Fits isotonic on train window, evaluates on test.
    Without historical odds, mode=model_only (Brier/ECE/hit_rate; no ROI).
    """
    matches = extract_finished_matches(archives, years=years)
    if len(matches) < 10:
        return BacktestMetrics(
            mode="model_only",
            sample_size=0,
            brier_1x2=0,
            ece_over=0,
            ece_under=0,
            ece_btts=0,
            hit_rate_1x2=0,
            windows=0,
            details={"error": "insufficient matches"},
        )

    agg = {
        "probs_1x2": [],
        "labels_1x2": [],
        "p_over": [],
        "y_over": [],
        "p_under": [],
        "y_under": [],
        "p_btts": [],
        "y_btts": [],
        "hit_correct": 0,
        "hit_total": 0,
    }
    windows = 0

    for train, test in iter_walk_forward_windows(matches, train_size, test_size):
        windows += 1
        calibrators = _fit_calibrators(archives, train) if apply_calibration else None
        chunk = _collect_predictions(archives, test, calibrators)
        agg["probs_1x2"].extend(chunk["probs_1x2"])
        agg["labels_1x2"].extend(chunk["labels_1x2"])
        agg["p_over"].extend(chunk["p_over"])
        agg["y_over"].extend(chunk["y_over"])
        agg["p_under"].extend(chunk["p_under"])
        agg["y_under"].extend(chunk["y_under"])
        agg["p_btts"].extend(chunk["p_btts"])
        agg["y_btts"].extend(chunk["y_btts"])
        agg["hit_correct"] += chunk["hit_correct"]
        agg["hit_total"] += chunk["hit_total"]

    all_test_matches = []
    for train, test in iter_walk_forward_windows(matches, train_size, test_size):
        all_test_matches.extend(test)

    roi, max_dd, roi_details = simulate_roi_flat_ev(archives, all_test_matches, db=db)
    mode = "roi_sim" if roi is not None else "model_only"

    return _aggregate_metrics(
        agg["probs_1x2"],
        agg["labels_1x2"],
        agg["p_over"],
        agg["y_over"],
        agg["p_under"],
        agg["y_under"],
        agg["p_btts"],
        agg["y_btts"],
        agg["hit_correct"],
        agg["hit_total"],
        mode=mode,
        windows=windows,
        roi_sim=roi,
        max_drawdown=max_dd,
        roi_details=roi_details,
    )


def run_holdout_backtest(
    archives: dict[int, dict],
    *,
    train_years: list[int] | None = None,
    test_years: list[int] | None = None,
    db=None,
) -> BacktestMetrics:
    """Train calibrators on train_years, evaluate raw + calibrated on test_years."""
    train_years = train_years or [2018]
    test_years = test_years or [2022]
    train_matches = extract_finished_matches(archives, years=train_years)
    test_matches = extract_finished_matches(archives, years=test_years)

    if not test_matches:
        return BacktestMetrics(
            mode="model_only",
            sample_size=0,
            brier_1x2=0,
            ece_over=0,
            ece_under=0,
            ece_btts=0,
            hit_rate_1x2=0,
            details={"error": "no test matches"},
        )

    calibrators = _fit_calibrators(archives, train_matches) if train_matches else None
    chunk = _collect_predictions(archives, test_matches, calibrators)
    roi, max_dd, roi_details = simulate_roi_flat_ev(archives, test_matches, db=db)
    mode = "roi_sim" if roi is not None else "model_only"
    return _aggregate_metrics(
        chunk["probs_1x2"],
        chunk["labels_1x2"],
        chunk["p_over"],
        chunk["y_over"],
        chunk["p_under"],
        chunk["y_under"],
        chunk["p_btts"],
        chunk["y_btts"],
        chunk["hit_correct"],
        chunk["hit_total"],
        mode=mode,
        windows=1,
        roi_sim=roi,
        max_drawdown=max_dd,
        roi_details=roi_details,
    )
