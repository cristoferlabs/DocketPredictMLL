#!/usr/bin/env python3
"""Fit isotonic calibration from WC archives (local, no Redis)."""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.api.services.worldcup_engine import set_calibration_factors
from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
from apps.worker.ml.calibration import fit_calibration_bundle


async def main() -> None:
    archives = await fetch_all_worldcup_archives()
    factors, _cals, metrics = fit_calibration_bundle(archives, train_years=[2018, 2022])
    set_calibration_factors(factors)
    print(json.dumps({"factors": factors, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
