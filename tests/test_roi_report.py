"""Tests /roi report formatter."""

from apps.api.services.stats_report import build_roi_report


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


def test_build_roi_report_includes_header(monkeypatch):
    monkeypatch.setattr(
        "apps.api.services.stats_report.evaluate_engine_health",
        lambda *a, **k: type("H", (), {"status": "ok", "alerts": []})(),
    )
    monkeypatch.setattr(
        "apps.api.services.stats_report.simulate_live_roi_from_db",
        lambda *a, **k: type(
            "R", (), {"bets": 0, "roi": None, "hit_rate": None, "max_drawdown": 0, "skipped_no_odds": 0}
        )(),
    )
    db = _FakeDb({"wc_predictions": _FakeQuery([]), "model_performance_metrics": _FakeQuery([])})
    text = build_roi_report(db)
    assert "ROI — Motor WC 2026" in text
    assert "SHARP" in text
    assert "Backtest histórico" in text
