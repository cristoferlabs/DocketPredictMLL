#!/usr/bin/env python3
"""Run WC walk-forward backtest locally (no Redis required)."""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
from apps.worker.ml.backtest import run_holdout_backtest, run_walk_forward_backtest


async def main() -> None:
    archives = await fetch_all_worldcup_archives()
    walk = run_walk_forward_backtest(archives, years=[2018, 2022])
    holdout = run_holdout_backtest(archives, train_years=[2018], test_years=[2022])
    print(
        json.dumps(
            {"walk_forward": walk.to_dict(), "holdout": holdout.to_dict()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
