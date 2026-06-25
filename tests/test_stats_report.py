"""Tests for pro stats report."""

from apps.api.services.stats_report import aggregate_wc_predictions, build_pro_stats_report
from apps.worker.ml.model_learning import LearningState


class _FakeResult:
    def __init__(self, data=None, count=None):
        self.data = data or []
        self.count = count


class _FakeQuery:
    def __init__(self, data=None, count=None):
        self._data = data or []
        self._count = count

    def select(self, *args, **kwargs):
        return self

    def order(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def is_(self, *args, **kwargs):
        return self

    def not_(self):
        return self

    def execute(self):
        return _FakeResult(self._data, self._count)


class _FakeSchema:
    def __init__(self, tables: dict):
        self._tables = tables

    def table(self, name):
        return self._tables[name]


class _FakeDb:
    def __init__(self, tables: dict):
        self._tables = tables

    def schema(self, name):
        return _FakeSchema(self._tables)


def test_build_pro_stats_report_includes_weights_and_learning(monkeypatch):
    monkeypatch.setattr(
        "apps.api.services.stats_report.load_learning_state",
        lambda: LearningState(n_updates=5, rolling_brier_n=3, rolling_brier_sum=1.5),
    )
    monkeypatch.setattr(
        "apps.api.services.stats_report.evaluate_live_brier_from_db",
        lambda db: None,
    )
    db = _FakeDb(
        {
            "wc_predictions": _FakeQuery(
                data=[
                    {
                        "evaluated_at": "2026-06-01",
                        "is_correct": True,
                        "brier_score": 0.2,
                        "metadata": {"sharp_tier": "A", "clv": {"clv_vs_close": 0.03}},
                    },
                    {
                        "evaluated_at": None,
                        "metadata": {"sharp_tier": "B"},
                    },
                ]
            ),
            "model_performance_metrics": _FakeQuery(data=[]),
            "odds_snapshots": _FakeQuery(count=2),
        }
    )
    msg = build_pro_stats_report(db)
    assert "STATS PRO" in msg
    assert "Poisson" in msg
    assert "Learning loop" in msg


def test_aggregate_wc_predictions_tier_hit_rate():
    db = _FakeDb(
        {
            "wc_predictions": _FakeQuery(
                data=[
                    {
                        "evaluated_at": "x",
                        "is_correct": True,
                        "brier_score": 0.1,
                        "metadata": {"sharp_tier": "A"},
                    },
                    {
                        "evaluated_at": "x",
                        "is_correct": False,
                        "brier_score": 0.3,
                        "metadata": {"sharp_tier": "A"},
                    },
                ]
            ),
        }
    )
    stats = aggregate_wc_predictions(db)
    assert stats.evaluated == 2
    assert stats.hit_rate == 0.5
    assert stats.tier_stats["A"]["hit_rate"] == 0.5
