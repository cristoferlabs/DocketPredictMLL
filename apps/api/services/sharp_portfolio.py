"""
SHARP v2 — portfolio ranking mode (quant desk).

De filtro binario → ranking por percentil del día + score compuesto.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.api.services.parlay_engine import SharpParlayPick

SOFT_REJECT_PREFIXES = (
    "confidence ",
    "mds ",
)


@dataclass(frozen=True)
class RankedSharpPick:
    pick: SharpParlayPick
    rank_score: float
    rank_position: int
    percentile: float
    tier: str  # A | B | C | X


def sharp_rank_score(
    *,
    ev_fair: float,
    confidence: float,
    mds: float,
) -> float:
    """Score compuesto para ordenar oportunidades (mayor = mejor)."""
    return round(
        0.45 * max(0.0, ev_fair) * 100.0
        + 0.30 * max(0.0, min(100.0, confidence))
        + 0.25 * max(0.0, min(100.0, mds)),
        4,
    )


def _is_soft_reject(reason: str | None) -> bool:
    if not reason:
        return False
    return any(reason.startswith(p) for p in SOFT_REJECT_PREFIXES)


def _is_hard_reject(reason: str | None) -> bool:
    if reason is None:
        return False
    if _is_soft_reject(reason):
        return False
    return True


def _tier_from_percentile(pct: float) -> str:
    if pct <= 0.20:
        return "A"
    if pct <= 0.45:
        return "B"
    if pct <= 0.75:
        return "C"
    return "X"


def rank_sharp_picks(
    picks: list[SharpParlayPick],
    *,
    top_pct: float = 0.30,
    top_k: int = 15,
) -> list[RankedSharpPick]:
    """Ordena picks por score; asigna tier por percentil."""
    candidates: list[tuple[SharpParlayPick, float]] = []
    for p in picks:
        if _is_hard_reject(p.reject_reason):
            continue
        if p.ev_fair <= 0 and p.reject_reason:
            continue
        score = sharp_rank_score(
            ev_fair=p.ev_fair,
            confidence=p.confidence,
            mds=p.mds,
        )
        candidates.append((p, score))

    candidates.sort(key=lambda x: (-x[1], -x[0].ev_fair, -x[0].mds))
    n = len(candidates)
    if n == 0:
        return []

    raw_promote = max(1, int(round(n * top_pct)))
    promote_n = max(min(2, n), min(top_k, raw_promote)) if n >= 2 else min(top_k, raw_promote)
    ranked: list[RankedSharpPick] = []
    for i, (p, score) in enumerate(candidates):
        pct = (i + 1) / n
        tier = _tier_from_percentile(pct) if i < promote_n else "X"
        if i < promote_n and tier == "X":
            tier = "C"
        ranked.append(
            RankedSharpPick(
                pick=p,
                rank_score=score,
                rank_position=i + 1,
                percentile=round(pct, 4),
                tier=tier,
            )
        )
    return ranked


def promote_portfolio_picks(
    picks: list[SharpParlayPick],
    *,
    top_pct: float = 0.30,
    top_k: int = 15,
) -> list[SharpParlayPick]:
    """
    Promueve top-K/percentil a elegibles para parlay.

    Solo revierte rechazos blandos (confidence/mds); mantiene hard blocks.
    """
    from apps.api.services.parlay_engine import SharpParlayPick as Pick
    ranked = rank_sharp_picks(picks, top_pct=top_pct, top_k=top_k)
    promoted_ids = {
        r.pick.match_id
        for r in ranked
        if r.tier in ("A", "B", "C")
    }

    out: list[SharpParlayPick] = []
    for p in picks:
        if p.match_id not in promoted_ids:
            out.append(p)
            continue
        if _is_hard_reject(p.reject_reason):
            out.append(p)
            continue
        out.append(
            Pick(
                match_id=p.match_id,
                team1=p.team1,
                team2=p.team2,
                fecha=p.fecha,
                ronda=p.ronda,
                outcome=p.outcome,
                market=p.market,
                p_model=p.p_model,
                odds=p.odds,
                ev_fair=p.ev_fair,
                confidence=p.confidence,
                mds=p.mds,
                correlation_group=p.correlation_group,
                reject_reason=None,
            )
        )
    return out


def portfolio_tier_for_confidence(confidence: int, *, settings=None) -> str:
    """Tier para singles en modo portfolio."""
    from apps.shared.config import get_settings

    settings = settings or get_settings()
    if confidence >= settings.sharp_min_composite:
        return "A"
    if confidence >= settings.sharp_portfolio_min_composite:
        return "B"
    if confidence >= settings.sharp_watch_composite:
        return "C"
    return "X"
