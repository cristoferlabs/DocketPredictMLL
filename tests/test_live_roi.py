"""Tests live ROI and engine health alerts."""

from apps.api.services.engine_health import evaluate_engine_health
from apps.worker.ml.live_roi import simulate_live_roi_from_db
from apps.worker.ml.model_learning import LearningState


class _FakeResult:
    def __init__(self, data=None, count=None):
        self.data = data or []
        self.count = count


class _FakeQuery:
    def __init__(self, data=None):
        self._data = data or []

    @property
    def not_(self):
        return self

    def select(self, *args, **kwargs):
        return self

    def is_(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def order(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def execute(self):
        return _FakeResult(self._data)


class _FakeSchema:
    def __init__(self, tables):
        self._tables = tables

    def table(self, name):
        return self._tables[name]


class _FakeDb:
    def __init__(self, tables):
        self._tables = tables

    def schema(self, name):
        return _FakeSchema(self._tables)


def test_simulate_live_roi_sharp_wins():
    db = _FakeDb(
        {
            "wc_predictions": _FakeQuery(
                [
                    {
                        "team_home": "A",
                        "team_away": "B",
                        "market_type": "1X2",
                        "predicted_outcome": "A",
                        "is_correct": True,
                        "expected_value_fair": 0.05,
                        "metadata": {
                            "sharp_allowed": True,
                            "clv": {"pick_odds": 2.0},
                        },
                        "evaluated_at": "2026-06-01",
                    },
                    {
                        "team_home": "C",
                        "team_away": "D",
                        "market_type": "1X2",
                        "predicted_outcome": "C",
                        "is_correct": False,
                        "expected_value_fair": 0.04,
                        "metadata": {"sharp_allowed": True, "clv": {"pick_odds": 1.8}},
                        "evaluated_at": "2026-06-02",
                    },
                ]
            ),
        }
    )
    r = simulate_live_roi_from_db(db, scope="sharp")
    assert r.bets == 2
    assert r.wins == 1
    assert r.roi == 0.0
    assert r.hit_rate == 0.5


def test_engine_health_critical_on_negative_clv(monkeypatch):
    monkeypatch.setattr(
        "apps.api.services.engine_health.load_learning_state",
        lambda: LearningState(
            rolling_clv_sum=-0.08,
            rolling_clv_n=6,
            rolling_brier_sum=2.0,
            rolling_brier_n=6,
        ),
    )
    monkeypatch.setattr(
        "apps.api.services.engine_health.simulate_live_roi_from_db",
        lambda *a, **k: type("R", (), {"bets": 0, "roi": None, "hit_rate": None})(),
    )
    health = evaluate_engine_health(_FakeDb({}))
    assert health.status == "critical"
    assert any("CLV" in a for a in health.alerts)
