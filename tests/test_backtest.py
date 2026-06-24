"""Backtest regression tests."""

from apps.worker.ml.backtest import run_holdout_backtest, run_walk_forward_backtest


def _mini_archive(year: int, matches: list[dict]) -> dict:
    return {"rounds": [{"name": "Group", "matches": matches}]}


def _synthetic_archives() -> dict[int, dict]:
    """Tiny WC-like dataset for offline tests."""
    m2018 = [
        {
            "date": "2018-06-14",
            "team1": {"name": "Russia"},
            "team2": {"name": "Saudi Arabia"},
            "score": {"ft": [5, 0]},
        },
        {
            "date": "2018-06-15",
            "team1": {"name": "Egypt"},
            "team2": {"name": "Uruguay"},
            "score": {"ft": [0, 1]},
        },
        {
            "date": "2018-06-15",
            "team1": {"name": "Portugal"},
            "team2": {"name": "Spain"},
            "score": {"ft": [3, 3]},
        },
    ]
    m2022 = [
        {
            "date": "2022-11-20",
            "team1": {"name": "Qatar"},
            "team2": {"name": "Ecuador"},
            "score": {"ft": [0, 2]},
        },
        {
            "date": "2022-11-21",
            "team1": {"name": "England"},
            "team2": {"name": "Iran"},
            "score": {"ft": [6, 2]},
        },
        {
            "date": "2022-11-22",
            "team1": {"name": "Argentina"},
            "team2": {"name": "Saudi Arabia"},
            "score": {"ft": [1, 2]},
        },
    ]
    return {2018: _mini_archive(2018, m2018), 2022: _mini_archive(2022, m2022)}


def test_walk_forward_produces_metrics():
    archives = _synthetic_archives()
    result = run_walk_forward_backtest(archives, train_size=2, test_size=1)
    assert result.sample_size >= 0
    assert result.mode == "model_only"
    assert result.roi_sim is None


def test_holdout_train_2018_test_2022():
    archives = _synthetic_archives()
    result = run_holdout_backtest(archives, train_years=[2018], test_years=[2022])
    assert result.sample_size == 3
    assert 0 <= result.brier_1x2 <= 2
    assert result.windows == 1
