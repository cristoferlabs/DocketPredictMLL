"""Structured audit log of discarded picks — written to artifacts/discard_log.jsonl.

Each line is a JSON object; append-only.  Not displayed to users; used for
post-hoc analysis of why the engine rejected apparently good opportunities.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from apps.api.services.telegram_terminal.betting_menu import MarketRow
    from apps.api.services.sharp_engine import SharpBetResult

_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "artifacts", "discard_log.jsonl"
)


# ── Reason parsers ─────────────────────────────────────────────────────────────

def _reasons_from_sharp(sharp_reason: str, tree_path: list[str]) -> list[str]:
    """Derive structured reason tags from the SHARP engine's human-readable reason."""
    r = sharp_reason.lower()
    path_str = " ".join(tree_path).lower()
    reasons: list[str] = []

    if "composite" in r or "confidence" in r:
        reasons.append("confidence_below_threshold")
    if "ev fair" in r and "<" in r:
        reasons.append("ev_below_threshold")
    if "tier" in r:
        reasons.append("portfolio_cut")
    if ("edge" in r or "ev máximo" in r) and ("pp" in r or "%" in r or "<" in r):
        reasons.append("edge_below_minimum")
    if "sin valor" in r or "no bet" in r or "sin señal" in r:
        reasons.append("no_bet_signal")
    if "watch" in r:
        reasons.append("downgraded_to_watch")
    if "cold" in path_str or "warm" in path_str or "mature" in path_str:
        reasons.append("phase_gate")
    if "outlier" in r or "outlier" in path_str:
        reasons.append("model_outlier")
    if "1x2" in r and ("δ" in r or "delta" in r or "pp" in r):
        reasons.append("1x2_divergence_block")
    if "mds" in r:
        reasons.append("mds_below_threshold")
    if not reasons:
        reasons.append("sharp_gate")
    return reasons


def _reasons_from_row(row: "MarketRow") -> list[str]:
    """Derive structured reason tags for a market row with no positive EV."""
    reasons: list[str] = []
    ev = row.ev_raw_pct if row.ev_raw_pct is not None else row.ev_pct
    if ev < 0:
        reasons.append("negative_ev")
    elif ev == 0:
        reasons.append("zero_ev")
    else:
        # ev_pct > 0 but ev_raw_pct <= 0 would be unusual — flag it
        reasons.append("ev_regime_capped_to_zero")
    return reasons


# ── Main entry ─────────────────────────────────────────────────────────────────

def log_discards(
    *,
    match: str,
    fecha: str | None,
    ronda: str | None,
    market_rows: list["MarketRow"],
    sharp: "SharpBetResult | None",
) -> None:
    """
    Append one JSON-line per discarded pick to artifacts/discard_log.jsonl.

    Discards are:
    1. market_rows with negative/zero EV and a real market quote (layer="market_rows")
    2. sharp engine decision if sharp_allowed=False (layer="sharp_gate")

    Silently swallows IO errors so it never disrupts the recommendation flow.
    """
    try:
        _write_discards(match=match, fecha=fecha, ronda=ronda,
                        market_rows=market_rows, sharp=sharp)
    except Exception:
        pass  # audit log must never break the main path


def _write_discards(
    *,
    match: str,
    fecha: str | None,
    ronda: str | None,
    market_rows: list["MarketRow"],
    sharp: "SharpBetResult | None",
) -> None:
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    entries: list[dict[str, Any]] = []

    # ── Layer 1: individual market rows with no positive expected value ────────
    for row in market_rows:
        if not row.has_market:
            continue  # no market quote → nothing to log
        ev_for_display = row.ev_raw_pct if row.ev_raw_pct is not None else row.ev_pct
        if ev_for_display > 0:
            continue  # positive EV → not a discard
        entries.append({
            "ts": ts,
            "match": match,
            "fecha": fecha,
            "ronda": ronda,
            "layer": "market_rows",
            "market": row.market_type,
            "selection": row.label,
            "model_prob": round(row.model_prob, 4),
            "market_odds": row.market_odds,
            "fair_odds": row.fair_odds,
            "market_implied": row.market_implied,
            "ev_decision": round(row.ev_pct, 2),
            "ev_raw": round(ev_for_display, 2),
            "discard_reasons": _reasons_from_row(row),
        })

    # ── Layer 2: sharp engine decision ────────────────────────────────────────
    if sharp is not None and not sharp.sharp_allowed:
        pick = sharp.decision.pick
        tree_path = sharp.decision.tree_path or []
        entries.append({
            "ts": ts,
            "match": match,
            "fecha": fecha,
            "ronda": ronda,
            "layer": "sharp_gate",
            "market": pick.market if pick else "n/d",
            "selection": pick.selection if pick else "n/d",
            "model_prob": round(pick.model_prob, 4) if pick else None,
            "market_odds": getattr(pick, "raw_odds", None) if pick else None,
            "ev_decision": round(sharp.ev_final * 100, 2),
            "mds": sharp.mds,
            "confidence": sharp.decision.confidence_score,
            "portfolio_tier": getattr(sharp, "portfolio_tier", None),
            "sharp_reason": sharp.sharp_reason,
            "soft_action": sharp.decision.soft_action,
            "discard_reasons": _reasons_from_sharp(sharp.sharp_reason, tree_path),
        })

    if not entries:
        return

    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
