"""Tests for favorite bias audit."""

from apps.worker.ml.favorite_bias_audit import (
    _match_row,
    aggregate_favorite_bias,
)


def test_favorite_compression_detected():
    row = _match_row(
        "Scotland",
        "Brazil",
        [
            ("Scotland", 0.235, 0.10),
            ("Empate", 0.262, 0.174),
            ("Brazil", 0.503, 0.758),
        ],
    )
    assert row is not None
    assert row.favorite_compression > 0.20
    assert row.underdog_inflation > 0.10

    audit = aggregate_favorite_bias([row])
    assert audit.favorite_bias_score > 0
    assert "favorite_compression" in audit.bias_detected


def test_aligned_low_bias():
    row = _match_row(
        "A",
        "B",
        [
            ("A", 0.45, 0.44),
            ("Empate", 0.28, 0.27),
            ("B", 0.27, 0.29),
        ],
    )
    audit = aggregate_favorite_bias([row])
    assert abs(audit.favorite_bias_score) < 0.15
